import _ from 'lodash'
import * as THREE from 'three'
import { LabelTypeName, ShapeTypeName } from '../../common/types'
import { makeLabel } from '../../functional/states'
import { LabelType, ShapeType, State } from '../../functional/types'
import { Vector3D } from '../../math/vector3d'
import { getColorById } from '../util'
import { Label3DList } from './label3d_list'
import { Shape3D } from './shape3d'

/** Convert string to label type name enum */
export function labelTypeFromString (type: string): LabelTypeName {
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
  protected _labelState: LabelType
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
  /** Shapes */
  protected _shapes: Shape3D[]
  /** Set flag when editing */
  protected _editing: boolean

  constructor (labelList: Label3DList) {
    this._labelState = makeLabel()
    this._selected = false
    this._highlighted = false
    this._color = [0, 0, 0, 1]
    this._parent = null
    this._children = []
    this._temporary = false
    this._labelList = labelList
    this._shapes = []
    this._editing = false
  }

  /** Get label list */
  public get labelList (): Readonly<Label3DList> {
    return this._labelList
  }

  /** get label id */
  public get labelId (): number {
    return this._labelState.id
  }

  /** get track id */
  public get trackId (): number {
    return this._labelState.track
  }

  /** get item index */
  public get item (): number {
    return this._labelState.item
  }

  /** get label type */
  public get type (): string {
    return labelTypeFromString(this._labelState.type)
  }

  /** get whether label was manually drawn */
  public get manual (): boolean {
    return this._labelState.manual
  }

  /** set whether label was manually drawn */
  public setManual () {
    this._labelState.manual = true
  }

  /** Get whether label is being edited */
  public get editing (): boolean {
    return this._editing
  }

  /** Get whether label is being edited */
  public set editing (e: boolean) {
    this._editing = e
  }

  /** get label state */
  public get labelState (): Readonly<LabelType> {
    if (!this._labelState) {
      throw new Error('Label uninitialized')
    }
    return this._labelState
  }

  /** Get parent label */
  public get parent (): Label3D | null {
    return this._parent
  }

  /** Set parent label */
  public set parent (parent: Label3D | null) {
    this._parent = parent
    if (parent && this._labelState) {
      this._labelState.parent = parent.labelId
    } else if (this._labelState) {
      this._labelState.parent = -1
    }
  }

  /** Get children */
  public get children (): Readonly<Label3D[]> {
    return this._children
  }

  /** Returns true if any children selected */
  public anyChildSelected (): boolean {
    for (const child of this.children) {
      if (child.selected) {
        return true
      }
    }

    return false
  }

  /** select the label */
  public set selected (s: boolean) {
    this._selected = s
  }

  /** return whether label selected */
  public get selected (): boolean {
    return this._selected
  }

  /** Return whether this label is temporary (not committed to state) */
  public get temporary (): boolean {
    return this._temporary
  }

  /** Get shape id's and shapes for updating */
  public abstract shapeStates (): [number[], ShapeTypeName[], ShapeType[]]

  /** highlight the label */
  public setHighlighted (intersection?: THREE.Intersection) {
    if (intersection) {
      this._highlighted = true
    } else {
      this._highlighted = false
    }
  }

  /** add child */
  public addChild (child: Label3D) {
    if (child.parent !== this) {
      if (child.parent) {
        child.parent.removeChild(child)
      }
      this._children.push(child)
      child.parent = this
      if (this._labelState) {
        this._labelState.children.push(child.labelId)
      }
    }
  }

  /** remove child */
  public removeChild (child: Label3D) {
    const index = this._children.indexOf(child)
    if (index >= 0) {
      this._children.splice(index, 1)
      child.parent = null
      if (this._labelState) {
        const stateIndex = this._labelState.children.indexOf(child.labelId)
        if (stateIndex >= 0) {
          this._labelState.children.splice(stateIndex, 1)
        }
      }
    }
  }

  /** get category */
  public get category (): number[] {
    if (this._labelState && this._labelState.category) {
      return this._labelState.category
    }
    return []
  }

  /** get attributes */
  public get attributes (): {[key: number]: number[]} {
    if (this._labelState && this._labelState.attributes) {
      return this._labelState.attributes
    }
    return {}
  }

  /** Set active camera for label */
  public set activeCamera (_camera: THREE.Camera) {
    return
  }

  /**
   * Handle mouse move
   * @param projection
   */
  public abstract onMouseDown (
    x: number, y: number, camera: THREE.Camera
  ): boolean

  /**
   * Handle mouse up
   * @param projection
   */
  public abstract onMouseUp (): void

  /**
   * Handle mouse move
   * @param projection
   */
  public abstract onMouseMove (
    x: number, y: number, camera: THREE.Camera
  ): boolean

  /** Rotate label in direction of quaternion */
  public abstract rotate (
    quaternion: THREE.Quaternion,
    anchor?: THREE.Vector3
  ): void

  /** Translate label in provided direction */
  public abstract translate (delta: THREE.Vector3): void

  /** Scale label */
  public abstract scale (
    scale: THREE.Vector3, anchor: THREE.Vector3, local: boolean
  ): void

  /** Move label to position, different from translate, which accepts a delta */
  public abstract move (position: THREE.Vector3): void

  /** Center of label */
  public get center (): THREE.Vector3 {
    return new THREE.Vector3()
  }

  /** Orientation of label */
  public get orientation (): THREE.Quaternion {
    return new THREE.Quaternion()
  }

  /** Size of the label */
  public get size (): THREE.Vector3 {
    return new THREE.Vector3()
  }

  /** Bounds of label */
  public bounds (_local?: boolean): THREE.Box3 {
    return new THREE.Box3()
  }

  /**
   * Initialize label
   * @param {State} state
   */
  public abstract init (
    itemIndex: number,
    category: number,
    center?: Vector3D,
    sensors?: number[],
    temporary?: boolean
  ): void

  /**
   * Return a list of the shape for inspection and testing
   */
  public abstract shapes (): Shape3D[]

  /** Convert label state to drawable */
  public updateState (
    state: State,
    itemIndex: number,
    labelId: number
  ): void {
    const item = state.task.items[itemIndex]
    this._labelState = _.cloneDeep(item.labels[labelId])
    this._color = getColorById(this.labelId, this.trackId)
    const select = state.user.select
    if (this._labelState.item in select.labels &&
        select.labels[this._labelState.item].includes(labelId)) {
      this.selected = true
    } else {
      this.selected = false
    }
    this._shapes = []
    for (const shapeId of this._labelState.shapes) {
      const shape = this._labelList.getShape(shapeId)
      if (shape) {
        this._shapes.push(shape)
      } else {
        throw new Error(`Could not find shape with id ${shapeId}`)
      }
    }
  }
}

export default Label3D
