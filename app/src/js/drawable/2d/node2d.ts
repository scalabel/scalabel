import { ShapeTypeName } from '../../common/types'
import { makeIndexedShape } from '../../functional/states'
import { IndexedShapeType, Node2DType } from '../../functional/types'
import { Point2D } from './point2d'

/** Points for custom labels */
export class Node2D extends Point2D {
  /** Color of this node, if any */
  private _color?: number[]
  /** node state */
  private _nodeState: Node2DType

  constructor (node?: Node2DType) {
    if (node) {
      super(node.x, node.y)
      this._nodeState = node
    } else {
      super()
      this._nodeState = {
        name: '',
        x: 0,
        y: 0
      }
    }
    this._indexedShape =
      makeIndexedShape(-1, -1, [], ShapeTypeName.NODE_2D, this._nodeState)
  }

  /** Get type name */
  public get typeName () {
    return ShapeTypeName.NODE_2D
  }

  /** Set name */
  public set name (name: string) {
    this._nodeState.name = name
  }

  /** Get name */
  public get name (): string {
    return this._nodeState.name
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
    return Boolean(this._nodeState.hidden)
  }

  /** Hide node */
  public hide () {
    this._nodeState.hidden = true
  }

  /** Override update state */
  public updateState (indexedShape: IndexedShapeType) {
    super.updateState(indexedShape)
    this._nodeState = indexedShape.shape as Node2DType
  }

  /** To state */
  public toState (): IndexedShapeType {
    return {
      ...this._indexedShape,
      shape: {
        ...this._nodeState,
        x: this.x,
        y: this.y
      }
    }
  }
}
