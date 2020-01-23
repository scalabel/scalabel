import _ from 'lodash'
import Label2D from '../../drawable/2d/label2d'
import Label3D from '../../drawable/3d/label3d'
import { makeIndexedShape, makeTrack } from '../../functional/states'
import { IndexedShapeType, Label2DTemplateType, LabelType, State, TrackType } from '../../functional/types'
import { LabelTypeName, TrackPolicyType } from '../types'
import { Box2DLinearInterpolationPolicy } from './policy/linear_interpolation/box2d_linear_interpolation'
import { Box3DLinearInterpolationPolicy } from './policy/linear_interpolation/box3d_linear_interpolation'
import { CustomLabel2DLinearInterpolationPolicy } from './policy/linear_interpolation/custom_label2d_linear_interpolation'
import { Plane3DLinearInterpolationPolicy } from './policy/linear_interpolation/plane3d_linear_interpolation'
import { PolygonLinearInterpolationPolicy } from './policy/linear_interpolation/polygon_linear_interpolation'
import { TrackPolicy } from './policy/policy'

export type Label = Label2D | Label3D

/** Convert policy type name to enum */
export function policyFromString (
  typeName: string
): TrackPolicyType {
  switch (typeName) {
    case TrackPolicyType.LINEAR_INTERPOLATION:
      return TrackPolicyType.LINEAR_INTERPOLATION
    case TrackPolicyType.NONE:
      return TrackPolicyType.NONE
  }

  throw new Error(`Unrecognized policy type: ${typeName}`)
}

/** Returns a function for creating a policy object based on the track type */
export function policyFactoryMaker (policyType: TrackPolicyType): (
  track: Track,
  type: string,
  label2DTemplates: {[name: string]: Label2DTemplateType}
) => TrackPolicy {
  switch (policyType) {
    case TrackPolicyType.NONE:
      return (track: Track) => new TrackPolicy(track)
    case TrackPolicyType.LINEAR_INTERPOLATION:
      return linearInterpolationPolicyFactory
  }
  throw new Error(`Invalid policy type ${policyType}`)
}

/** Factory for linear interpolation policies */
export function linearInterpolationPolicyFactory (
  track: Track,
  type: string,
  label2DTemplates: {[name: string]: Label2DTemplateType}
): TrackPolicy {
  switch (type) {
    case LabelTypeName.BOX_2D:
      return new Box2DLinearInterpolationPolicy(track)
    case LabelTypeName.BOX_3D:
      return new Box3DLinearInterpolationPolicy(track)
    case LabelTypeName.PLANE_3D:
      return new Plane3DLinearInterpolationPolicy(track)
    case LabelTypeName.POLYGON_2D:
    case LabelTypeName.POLYLINE_2D:
      return new PolygonLinearInterpolationPolicy(track)
  }
  if (type in label2DTemplates) {
    return new CustomLabel2DLinearInterpolationPolicy(track)
  }
  throw new Error(
    `Linear interpolation is not supported for track type ${type}`
  )
}

/**
 * Object representation of track
 */
export class Track {
  /** policy */
  protected _policy: TrackPolicy
  /** track state */
  protected _track: TrackType
  /** shape map */
  protected _shapes: { [index: number]: IndexedShapeType[] }
  /** label map */
  protected _labels: { [index: number]: LabelType }
  /** updated indices */
  protected _updatedIndices: Set<number>
  /** type */
  protected _type: string

  constructor () {
    this._track = makeTrack(-1, LabelTypeName.EMPTY)
    this._policy = new TrackPolicy(this)
    this._shapes = {}
    this._labels = {}
    this._updatedIndices = new Set()
    this._type = ''
  }

  /**
   * Run when state is updated
   * @param state
   */
  public updateState (state: State, id: number) {
    this._track = state.task.tracks[id]
    const policyType = policyFromString(
      state.task.config.policyTypes[state.user.select.policyType]
    )
    if (policyType !== this.policyType) {
      this._policy = policyFactoryMaker(policyType)(
        this,
        this._track.type,
        state.task.config.label2DTemplates
      )
    }
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
    return this._policy.type
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

  /** Get indices where the track has labels */
  public get indices (): number[] {
    return Object.keys(this._labels).map(
      (key) => Number(key)
    ).sort(
      (a, b) => a - b
    )
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

  /** Set shapes at item index */
  public setShapes (
    index: number,
    shapes: IndexedShapeType[]
  ) {
    this._updatedIndices.add(index)
    this._shapes[index] = shapes
  }

  /** Add updated index */
  public addUpdatedIndex (index: number) {
    this._updatedIndices.add(index)
  }

  /** Get updated indices */
  public get updatedIndices (): Readonly<number[]> {
    return Array.from(this._updatedIndices)
  }

  /** Clear updated indices */
  public clearUpdatedIndices () {
    this._updatedIndices.clear()
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
        this._updatedIndices.add(index)
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
      this._updatedIndices.add(itemIndex)
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

      this._policy.update(itemIndex)
    }
  }
}
