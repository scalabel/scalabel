import _ from "lodash"
import * as THREE from "three"

import { LabelTypeName } from "../../const/common"
import { makeLabel } from "../../functional/states"
import { Vector3D } from "../../math/vector3d"
import {
  IdType,
  INVALID_ID,
  LabelType,
  ShapeType,
  State
} from "../../types/state"
import { getColorById } from "../util"
import { Label3DList } from "./label3d_list"
import { Shape3D } from "./shape3d"

/**
 * Convert string to label type name enum
 *
 * @param type
 */
export function labelTypeFromString(type: string): LabelTypeName {
  switch (type) {
    case LabelTypeName.BOX_3D:
      return LabelTypeName.BOX_3D
    case LabelTypeName.PLANE_3D:
      return LabelTypeName.PLANE_3D
  }
  return LabelTypeName.EMPTY
}

/**
 * Abstract class for 3D drawable labels
 */
export abstract class Label3D {
  /** the corresponding label in the state */
  protected _label: LabelType
  /** whether the label is selected */
  protected _selected: boolean
  /** whether the label is highlighted */
  protected _highlighted: boolean
  /** rgba color decided by labelId */
  protected _color: number[]
  /** parent label if any */
  protected _parent: Label3D | null
  /** children if any */
  protected _children: Label3D[]
  /** Whether this is temporary */
  protected _temporary: boolean
  /** label list this belongs to */
  protected _labelList: Label3DList

  /**
   * Constructor
   *
   * @param labelList
   */
  constructor(labelList: Label3DList) {
    this._label = makeLabel()
    this._selected = false
    this._highlighted = false
    this._color = [0, 0, 0, 1]
    this._parent = null
    this._children = []
    this._temporary = true
    this._labelList = labelList
  }

  /** Get label list */
  public get labelList(): Readonly<Label3DList> {
    return this._labelList
  }

  /** get label id */
  public get labelId(): IdType {
    return this._label.id
  }

  /** get track id */
  public get trackId(): IdType {
    return this._label.track
  }

  /** get item index */
  public get item(): number {
    return this._label.item
  }

  /** get label type */
  public get type(): string {
    return labelTypeFromString(this._label.type)
  }

  /** get whether label was manually drawn */
  public get manual(): boolean {
    return this._label.manual
  }

  /** set whether label was manually drawn */
  public setManual(): void {
    this._label.manual = true
  }

  /** get label state */
  public get label(): Readonly<LabelType> {
    return this._label
  }

  /** Get parent label */
  public get parent(): Label3D | null {
    return this._parent
  }

  /** Set parent label */
  public set parent(parent: Label3D | null) {
    this._parent = parent
    if (parent !== null) {
      this._label.parent = parent.labelId
    } else {
      this._label.parent = INVALID_ID
    }
  }

  /** Get children */
  public get children(): Readonly<Label3D[]> {
    return this._children
  }

  /** Returns true if any children selected */
  public anyChildSelected(): boolean {
    for (const child of this.children) {
      if (child.selected) {
        return true
      }
    }

    return false
  }

  /** select the label */
  public set selected(s: boolean) {
    this._selected = s
  }

  /** return whether label selected */
  public get selected(): boolean {
    return this._selected
  }

  /** Return whether this label is temporary (not committed to state) */
  public get temporary(): boolean {
    return this._temporary
  }

  /** Get shape id's and shapes for updating */
  public abstract shapes(): ShapeType[]

  /**
   * highlight the label
   *
   * @param intersection
   */
  public setHighlighted(intersection?: THREE.Intersection): void {
    if (intersection !== undefined) {
      this._highlighted = true
    } else {
      this._highlighted = false
    }
  }

  /**
   * add child
   *
   * @param child
   */
  public addChild(child: Label3D): void {
    if (child.parent !== this) {
      if (child.parent !== null) {
        child.parent.removeChild(child)
      }
      this._children.push(child)
      child.parent = this
      this._label.children.push(child.labelId)
    }
  }

  /**
   * remove child
   *
   * @param child
   */
  public removeChild(child: Label3D): void {
    const index = this._children.indexOf(child)
    if (index >= 0) {
      this._children.splice(index, 1)
      child.parent = null
      const stateIndex = this._label.children.indexOf(child.labelId)
      if (stateIndex >= 0) {
        this._label.children.splice(stateIndex, 1)
      }
    }
  }

  /** get category */
  public get category(): number[] {
    return this._label.category
  }

  /** get attributes */
  public get attributes(): { [key: number]: number[] } {
    return this._label.attributes
  }

  /** Set active camera for label */
  // TODO: is this still useful?
  // eslint-disable-next-line accessor-pairs,require-jsdoc
  public set activeCamera(_camera: THREE.Camera) {}

  /**
   * Handle mouse move
   *
   * @param projection
   */
  public abstract onMouseDown(
    x: number,
    y: number,
    camera: THREE.Camera
  ): boolean

  /**
   * Handle mouse up
   *
   * @param projection
   */
  public abstract onMouseUp(): void

  /**
   * Handle mouse move
   *
   * @param projection
   */
  public abstract onMouseMove(
    x: number,
    y: number,
    camera: THREE.Camera
  ): boolean

  /** Rotate label in direction of quaternion */
  public abstract rotate(
    quaternion: THREE.Quaternion,
    anchor?: THREE.Vector3
  ): void

  /** Translate label in provided direction */
  public abstract translate(delta: THREE.Vector3): void

  /** Scale label */
  public abstract scale(
    scale: THREE.Vector3,
    anchor: THREE.Vector3,
    local: boolean
  ): void

  /** Move label to position, different from translate, which accepts a delta */
  public abstract move(position: THREE.Vector3): void

  /** Center of label */
  public get center(): THREE.Vector3 {
    return new THREE.Vector3()
  }

  /** Orientation of label */
  public get orientation(): THREE.Quaternion {
    return new THREE.Quaternion()
  }

  /** Size of the label */
  public get size(): THREE.Vector3 {
    return new THREE.Vector3()
  }

  /**
   * Bounds of label
   *
   * @param _local
   */
  public bounds(_local?: boolean): THREE.Box3 {
    return new THREE.Box3()
  }

  /**
   * Initialize label
   *
   * @param {State} state
   */
  public abstract init(
    itemIndex: number,
    category: number,
    center?: Vector3D,
    sensors?: number[],
    temporary?: boolean
  ): void

  /**
   * Return a list of the shape for inspection and testing
   */
  public abstract internalShapes(): Shape3D[]

  /**
   * Convert label state to drawable
   *
   * @param state
   * @param itemIndex
   * @param labelId
   */
  public updateState(state: State, itemIndex: number, labelId: IdType): void {
    const item = state.task.items[itemIndex]
    this._label = _.cloneDeep(item.labels[labelId])
    this._color = getColorById(this.labelId, this.trackId)
    const select = state.user.select
    if (
      this._label.item in select.labels &&
      select.labels[this._label.item].includes(labelId)
    ) {
      this.selected = true
    } else {
      this.selected = false
    }
  }
}

export default Label3D
