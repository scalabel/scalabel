import { makeNode2d } from "../../functional/states"
import { Node2DType } from "../../types/state"
import { Point2D } from "./point2d"

/** Points for custom labels */
export class Node2D extends Point2D {
  /** Name of this node */
  private _name: string
  /** Set if hidden */
  private _hidden: boolean
  /** Color of this node, if any */
  private readonly _color?: number[]

  /**
   * Constructor
   *
   * @param node
   */
  constructor(node: Node2DType) {
    super(node.x, node.y)
    this._name = node.name
    this._hidden = Boolean(node.hidden)
  }

  /** Set name */
  public set name(name: string) {
    this._name = name
  }

  /** Get name */
  public get name(): string {
    return this._name
  }

  /** Get color */
  public get color(): Readonly<number[]> | null {
    if (this._color !== null && this._color !== undefined) {
      return this._color
    }
    return null
  }

  /** Returns true if hidden */
  public get hidden(): boolean {
    return this._hidden
  }

  /** Hide node */
  public hide(): void {
    this._hidden = true
  }

  /** To state representation */
  public toState(): Node2DType {
    return makeNode2d({
      name: this._name,
      hidden: this._hidden,
      x: this.x,
      y: this.y
    })
  }
}
