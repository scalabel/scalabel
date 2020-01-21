import { ShapeTypeName } from '../../common/types'
import { Node2DType } from '../../functional/types'
import { Point2D } from './point2d'

/** Points for custom labels */
export class Node2D extends Point2D {
  /** Name of this node */
  private _name: string
  /** Set if hidden */
  private _hidden: boolean
  /** Color of this node, if any */
  private _color?: number[]

  constructor (node: Node2DType) {
    super(node.x, node.y)
    this._name = node.name
    this._hidden = Boolean(node.hidden)
  }

  /** Get type name */
  public get typeName () {
    return ShapeTypeName.NODE_2D
  }

  /** Convert to state */
  public toState (): Node2DType {
    return {
      x: this.x,
      y: this.y,
      name: this._name,
      hidden: this._hidden,
      color: this._color
    }
  }

  /** Set name */
  public set name (name: string) {
    this._name = name
  }

  /** Get name */
  public get name (): string {
    return this._name
  }

  /** Get color */
  public get color (): Readonly<number[]> | null {
    if (this._color) {
      return this._color
    }
    return null
  }

  /** Returns true if hidden */
  public get hidden (): boolean {
    return this._hidden
  }

  /** Hide node */
  public hide () {
    this._hidden = true
  }
}
