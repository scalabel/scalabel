import { Key, LabelTypeName } from "../../const/common"
import { Label2D, Label2DModifier } from "./label2d"
import { Polygon2D } from "./polygon2d"
import { PathPoint2D } from "./path_point2d"
import { PathPointType } from "../../types/state"

/**
 * Polygon2DBoundaryCloner is a 2d label modifier for polygons.
 * It clone a segment of the boundary of some other polygon to the target one.
 */
export class Polygon2DBoundaryCloner extends Label2DModifier {
  private _target: Polygon2D
  private readonly _initialPoints: PathPoint2D[]
  private _label: Polygon2D | undefined
  private _handler1Idx: number | undefined
  private _handler2Idx: number | undefined
  private _reversed: boolean
  private _finishCallback: (() => void) | undefined

  /**
   * Constructor
   *
   * @param target: the label to append the cloned boundary segment.
   * @param target
   */
  constructor(target: Label2D) {
    super()

    this._target = target as Polygon2D
    this._initialPoints = [...this._target.points]
    this._reversed = false
  }

  /**
   * Implementation of the `onClickHandler` abstract method.
   *
   * @param label
   * @param handlerIdx
   */
  public onClickHandler(label: Label2D, handlerIdx: number): void {
    if (label.type !== LabelTypeName.POLYGON_2D) {
      console.warn(`attempt to clone boundary on ${label.type}`)
      return
    }

    const l = label as Polygon2D
    if (l.points[handlerIdx - 1].type !== PathPointType.LINE) {
      // Can only select line point
      return
    }

    if (
      this._label === undefined ||
      this._handler1Idx === undefined ||
      this._label.labelId !== label.labelId
    ) {
      // Set current handler to be the initial vertex of the boundary segment
      // if no one is set yet or the current label is different from the
      // previously select one.
      this._label = label as Polygon2D
      this._handler1Idx = handlerIdx
      this._reversed = false
      return
    }

    // Set current handler to be the second vertex
    this._handler2Idx = handlerIdx
    this.updateRender()
  }

  /**
   * Implementation of the `onKeyDown` abstract method.
   *
   * @param e
   */
  public onKeyDown(e: KeyboardEvent): void {
    switch (e.key) {
      case Key.ALT:
        this._reversed = !this._reversed
        this.updateRender()
        break
      case Key.ENTER:
        this._finishCallback?.()
        break
      case Key.ESCAPE:
        this._target.points = [...this._initialPoints]
        this._finishCallback?.()
        break
    }
  }

  /**
   * Implementation of the `onFinish` abstract method.
   *
   * @param fn
   */
  public onFinish(fn: () => void): void {
    this._finishCallback = fn
  }

  /**
   * Update the rendering of the target for preview purpose
   */
  private updateRender(): void {
    const {
      _label: source,
      _reversed: reversed,
      _handler1Idx: h1,
      _handler2Idx: h2
    } = this
    if (source === undefined || h1 === undefined || h2 === undefined) {
      return
    }

    const qs = source.points
    const ps = [...this._initialPoints]
    const advance = reversed
      ? (i: number) => ((i - 2 + qs.length) % qs.length) + 1
      : (i: number) => (i % qs.length) + 1
    for (let idx = h1; idx !== h2; idx = advance(idx)) {
      const q = qs[idx - 1]
      ps.push(q)
    }
    const q = qs[h2 - 1].clone()
    ps.push(q)

    this._target.points = ps
  }
}
