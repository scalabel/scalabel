import _ from 'lodash'
import Label2D from '../../drawable/2d/label2d'
import Label3D from '../../drawable/3d/label3d'
import { makeIndexedShape, makeTrack } from '../../functional/states'
import { IndexedShapeType, LabelType, State, TrackType } from '../../functional/types'
import { LabelTypeName, TrackPolicyType } from '../types'

export type Label = Label2D | Label3D

/** Convert policy type name to enum */
export function policyFromString (
  typeName: string
): TrackPolicyType {
  switch (typeName) {
    case TrackPolicyType.LINEAR_INTERPOLATION:
      return TrackPolicyType.LINEAR_INTERPOLATION
  }

  throw new Error(`Unrecognized policy type: ${typeName}`)
}

/**
 * Object representation of track
 */
export abstract class Track {
  /** policy */
  protected _policy: TrackPolicyType
  /** track state */
  protected _track: TrackType
  /** shape map */
  protected _shapes: { [index: number]: IndexedShapeType[] }
  /** label map */
  protected _labels: { [index: number]: LabelType }
  /** updated indices */
  protected _updatedIndices: number[]
  /** type */
  protected _type: string

  constructor () {
    this._track = makeTrack(-1, LabelTypeName.EMPTY)
    this._policy = TrackPolicyType.NONE
    this._shapes = {}
    this._labels = {}
    this._updatedIndices = []
    this._type = ''
  }

  /**
   * Run when state is updated
   * @param state
   */
  public updateState (state: State, id: number) {
    this._track = state.task.tracks[id]
    this._policy = policyFromString(
      state.task.config.policyTypes[state.user.select.policyType]
    )
    const items = Object.keys(this._track.labels).map((key) => Number(key))
    this._labels = Object.assign({}, _.pick(this._labels, items))
    this._shapes = Object.assign({}, _.pick(this._shapes, items))
    for (const item of items) {
      const labelId = this._track.labels[item]
      const label = state.task.items[item].labels[labelId]
      this._labels[item] = label
      this._shapes[item] = []
      for (const shapeId of label.shapes) {
        this._shapes[item].push(state.task.items[item].shapes[shapeId])
      }
    }
  }

  /**
   * Get track policy
   */
  public get policyType (): TrackPolicyType {
    return this._policy
  }

  /**
   * Get track id
   */
  public get id () {
    return this._track.id
  }

  /** Get track type */
  public get type () {
    return this._type
  }

  /** Get label at index */
  public getLabel (index: number): Readonly<LabelType> | null {
    if (index in this._labels) {
      return this._labels[index]
    }
    return null
  }

  /** Get shapes at item index */
  public getShapes (
    index: number
  ): Readonly<Array<Readonly<IndexedShapeType>>> {
    if (index in this._shapes) {
      return this._shapes[index]
    }
    return []
  }

  /** Get updated indices */
  public get updatedIndices (): Readonly<number[]> {
    return this._updatedIndices
  }

  /** Clear updated indices */
  public clearUpdatedIndices () {
    this._updatedIndices.length = 0
  }

  /**
   * Get newly created labels when creating track
   * @param itemIndex
   * @param label
   */
  public init (
    itemIndex: number,
    label: Readonly<Label>,
    numItems: number,
    parentTrack?: Track
  ): void {
    this._shapes = {}
    this._labels = {}
    this._type = label.type
    const labelState = label.label
    const [,shapeTypes, shapeStates] = label.shapeStates()
    for (let index = itemIndex; index < itemIndex + numItems; index ++) {
      const cloned = _.cloneDeep(labelState) as LabelType
      cloned.item = -1
      if (index > itemIndex) {
        cloned.manual = false
      }

      if (parentTrack) {
        const parentLabel = parentTrack.getLabel(index)
        if (parentLabel) {
          cloned.item = index
          cloned.parent = parentLabel.id
        }
      } else {
        cloned.item = index
      }

      if (cloned.item === index) {
        this._labels[index] = cloned
        this._shapes[index] = []
        for (let i = 0; i < shapeTypes.length; i++) {
          this._shapes[index].push(makeIndexedShape(
            -1, [-1], shapeTypes[i], _.cloneDeep(shapeStates[i])
          ))
        }
        this._updatedIndices.push(index)
      }
    }
  }

  /**
   * Callback for when a label in the track is updated
   * @param itemIndex
   * @param newShapes
   */
  public update (itemIndex: number, label: Readonly<Label>): void {
    const [ids, shapeTypes, newShapes] = label.shapeStates()
    if (
      itemIndex in this._shapes &&
      newShapes.length === this._shapes[itemIndex].length
    ) {
      this._updatedIndices.push(itemIndex)
      this._shapes[itemIndex].length = ids.length
      for (let i = 0; i < newShapes.length; i++) {
        this._shapes[itemIndex][i].id = ids[i]
        this._shapes[itemIndex][i].type = shapeTypes[i]
        this._shapes[itemIndex][i].shape = newShapes[i]
      }

      this._labels[itemIndex] = {
        ...this._labels[itemIndex],
        ...label.label
      }

      switch (this._policy) {
        case TrackPolicyType.LINEAR_INTERPOLATION:
          this.linearInterpolationUpdate(itemIndex)
      }
    }
  }

  /** Linear interpolate */
  protected linearInterpolationUpdate (updatedIndex: number): void {
    const itemIndices =
      Object.keys(this._labels).map((key) => Number(key)).sort(
        (a, b) => a - b
      )
    const itemArrayUpdatedIndex = itemIndices.indexOf(updatedIndex)

    if (itemArrayUpdatedIndex < 0) {
      return
    }

    let lastManualIndex = -1
    for (let i = itemArrayUpdatedIndex - 1; i >= 0; i -= 1) {
      const index = itemIndices[i]
      this._updatedIndices.push(i)
      if (index in this._labels) {
        const label = this._labels[index]
        if (label.manual) {
          lastManualIndex = i
          break
        }
      }
    }

    if (lastManualIndex >= 0) {
      for (let i = lastManualIndex; i < itemArrayUpdatedIndex; i++) {
        this.linearInterpolateIndex(
          itemIndices[i],
          itemIndices[lastManualIndex],
          itemIndices[itemArrayUpdatedIndex]
        )
      }
    } else {
      for (let i = 0; i < itemArrayUpdatedIndex; i++) {
        const index = itemIndices[i]
        for (
          let shapeIndex = 0;
          shapeIndex < this._shapes[updatedIndex].length;
          shapeIndex++
        ) {
          this._shapes[index][shapeIndex].shape =
            _.cloneDeep(this._shapes[updatedIndex][shapeIndex].shape)
          this._shapes[index][shapeIndex].type =
            this._shapes[updatedIndex][shapeIndex].type
        }
        this._shapes[index].length = this._shapes[updatedIndex].length
      }
    }

    // Go forward
    let nextManualIndex = -1
    for (let i = updatedIndex + 1; i < itemIndices.length; i += 1) {
      const index = itemIndices[i]
      this._updatedIndices.push(i)
      if (index in this._labels) {
        const label = this._labels[index]
        if (label.manual) {
          nextManualIndex = i
          break
        }
      }
    }

    if (nextManualIndex >= 0) {
      for (let i = itemArrayUpdatedIndex + 1; i <= nextManualIndex; i++) {
        this.linearInterpolateIndex(
          itemIndices[i],
          itemIndices[itemArrayUpdatedIndex],
          itemIndices[nextManualIndex]
        )
      }
    } else {
      for (let i = itemArrayUpdatedIndex + 1; i < itemIndices.length; i++) {
        const index = itemIndices[i]
        for (
          let shapeIndex = 0;
          shapeIndex < this._shapes[updatedIndex].length;
          shapeIndex++
        ) {
          this._shapes[index][shapeIndex].shape =
            _.cloneDeep(this._shapes[updatedIndex][shapeIndex].shape)
          this._shapes[index][shapeIndex].type =
            this._shapes[updatedIndex][shapeIndex].type
        }
        this._shapes[index].length = this._shapes[updatedIndex].length
      }
    }
  }

  /** Copy shape */

  /** Linear interpolate shapes at index */
  protected abstract linearInterpolateIndex (
    itemIndex: number,
    previousIndex: number,
    nextIndex: number
  ): void
}
