import _ from "lodash"

import { TrackInterp } from "../auto/track/interp/interp"
import { Box2DLinearInterp } from "../auto/track/interp/linear/box2d"
import { Points2DLinearInterp } from "../auto/track/interp/linear/points2d"
import { LabelTypeName, TrackPolicyType } from "../const/common"
import Label2D from "../drawable/2d/label2d"
import Label3D from "../drawable/3d/label3d"
import { makeLabel, makeShape, makeTrack } from "../functional/states"
import { IdType, LabelType, ShapeType, State, TrackType } from "../types/state"
export type Label = Label2D | Label3D

/**
 * Convert policy type name to enum
 *
 * @param typeName
 */
export function policyFromString(typeName: string): TrackPolicyType {
  switch (typeName) {
    case TrackPolicyType.LINEAR_INTERPOLATION:
      return TrackPolicyType.LINEAR_INTERPOLATION
    case TrackPolicyType.NONE:
      return TrackPolicyType.NONE
  }

  throw new Error(`Unrecognized policy type: ${typeName}`)
}

/**
 * Returns a function for creating a policy object based on the track type
 *
 * @param policyType
 */
function policyFactoryMaker(
  policyType: TrackPolicyType
): (type: string) => TrackInterp {
  switch (policyType) {
    case TrackPolicyType.NONE:
      return () => new TrackInterp()
    case TrackPolicyType.LINEAR_INTERPOLATION:
      return linearInterpolationPolicyFactory
    default:
      // Just in case
      // eslint-disable-next-line @typescript-eslint/restrict-template-expressions
      throw new Error(`Invalid policy type ${policyType}`)
  }
}

/**
 * Factory for linear interpolation policies
 *
 * @param type
 */
function linearInterpolationPolicyFactory(type: string): TrackInterp {
  switch (type) {
    case LabelTypeName.BOX_2D:
      return new Box2DLinearInterp()
    case LabelTypeName.POLYGON_2D:
    case LabelTypeName.POLYLINE_2D:
      return new Points2DLinearInterp()
    default:
      throw new Error(`Unknown policy type ${type}`)
  }
}

/**
 * Object representation of track
 */
export class Track {
  /** policy */
  protected _policy: TrackInterp
  /** track state */
  protected _track: TrackType
  /** shape map */
  protected _shapes: { [itemIndex: number]: ShapeType[] }
  /** label map */
  protected _labels: { [itemIndex: number]: LabelType }
  /** updated indices */
  protected _updatedIndices: Set<number>
  /** type */
  protected _type: string
  /** tracking policy type */
  protected _policyType: string

  /**
   * Constructor
   */
  constructor() {
    this._track = makeTrack({ type: LabelTypeName.EMPTY })
    this._policy = new TrackInterp()
    this._shapes = {}
    this._labels = {}
    this._updatedIndices = new Set()
    this._type = ""
    this._policyType = ""
  }

  /**
   * Run when state is updated
   *
   * @param state
   * @param id
   */
  public updateState(state: State, id: IdType): void {
    this._track = state.task.tracks[id]
    const policyType = policyFromString(
      state.task.config.policyTypes[state.user.select.policyType]
    )
    if (policyType !== this._policyType) {
      this._policyType = policyType
      this._policy = policyFactoryMaker(policyType)(this._track.type)
    }
    const items = _.keys(this._track.labels).map((key) => Number(key))
    this._labels = Object.assign({}, _.pick(this._labels, items))
    this._shapes = Object.assign({}, _.pick(this._shapes, items))
    for (const item of items) {
      const labelId = this._track.labels[item]
      const label = state.task.items[item].labels[labelId]
      this._labels[item] = label
      this._shapes[item] = label.shapes.map(
        (shapeId) => state.task.items[item].shapes[shapeId]
      )
    }
  }

  /**
   * Get track id
   */
  public get id(): IdType {
    return this._track.id
  }

  /** Get track type */
  public get type(): string {
    return this._type
  }

  /** Get indices where the track has labels */
  public get indices(): number[] {
    return Object.keys(this._labels)
      .map((key) => Number(key))
      .sort((a, b) => a - b)
  }

  /**
   * Get label at index
   *
   * @param index
   */
  public getLabel(index: number): Readonly<LabelType> | null {
    if (index in this._labels) {
      return this._labels[index]
    }
    return null
  }

  /**
   * Get shapes at item index
   *
   * @param index
   */
  public getShapes(index: number): Readonly<Array<Readonly<ShapeType>>> {
    if (index in this._shapes) {
      return this._shapes[index]
    }
    return []
  }

  /**
   * Set shapes at item index
   *
   * @param index
   * @param shapes
   */
  public setShapes(index: number, shapes: ShapeType[]): void {
    this._updatedIndices.add(index)
    this._shapes[index] = shapes
  }

  /**
   * Add updated index
   *
   * @param index
   */
  public addUpdatedIndex(index: number): void {
    this._updatedIndices.add(index)
  }

  /** Get updated indices */
  public get updatedIndices(): Readonly<number[]> {
    return Array.from(this._updatedIndices)
  }

  /** Clear updated indices */
  public clearUpdatedIndices(): void {
    this._updatedIndices.clear()
  }

  /**
   * Get newly created labels when creating track
   *
   * @param itemIndex
   * @param label
   * @param numItems
   * @param parentTrack
   */
  public init(
    itemIndex: number,
    label: Readonly<Label>,
    numItems: number,
    parentTrack?: Track
  ): void {
    this._shapes = {}
    this._labels = {}
    this._type = label.type
    const labelState = label.label
    const shapeStates = label.shapes()
    for (let index = itemIndex; index < itemIndex + numItems; index++) {
      const cloned = makeLabel(labelState, true)
      cloned.item = -1
      cloned.track = label.trackId
      if (index > itemIndex) {
        cloned.manual = false
      }

      if (parentTrack !== undefined) {
        const parentLabel = parentTrack.getLabel(index)
        if (parentLabel !== null) {
          cloned.item = index
          cloned.parent = parentLabel.id
        }
      } else {
        cloned.item = index
      }

      if (cloned.item === index) {
        cloned.shapes = []
        this._shapes[index] = []
        for (const shape of shapeStates) {
          const newShape = makeShape(shape.shapeType, shape)
          this._shapes[index].push(newShape)
          cloned.shapes.push(newShape.id)
        }
        this._labels[index] = cloned
        this._updatedIndices.add(index)
      }
    }
  }

  /**
   * Callback for when a label in the track is updated
   *
   * @param itemIndex
   * @param newShapes
   * @param label
   */
  public update(itemIndex: number, label: Readonly<Label>): void {
    const newShapes = label.shapes()
    if (
      itemIndex in this._shapes &&
      newShapes.length === this._shapes[itemIndex].length
    ) {
      this._updatedIndices.add(itemIndex)
      this._shapes[itemIndex].length = newShapes.length
      for (let i = 0; i < newShapes.length; i++) {
        this._shapes[itemIndex][i] = newShapes[i]
      }

      this._labels[itemIndex] = {
        ...this._labels[itemIndex],
        ...label.label
      }

      const itemIndices = _.keys(this._labels).map((k) => Number(k))
      const labels = itemIndices.map((i) => this._labels[i])
      const shapes = itemIndices.map((i) => this._shapes[i])
      const newAllShapes = this._policy.interp(
        this._labels[itemIndex],
        newShapes,
        labels,
        shapes
      )
      for (let i = 0; i < itemIndices.length; i += 1) {
        if (newAllShapes[i] !== shapes[i]) {
          this._updatedIndices.add(itemIndices[i])
          this._shapes[itemIndices[i]] = newAllShapes[i]
        }
      }
    }
  }
}
