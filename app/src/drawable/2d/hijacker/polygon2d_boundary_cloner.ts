import { Key } from "../../../const/common"
import { Label2D } from "../label2d"
import { Polygon2D } from "../polygon2d"
import { PathPoint2D } from "../path_point2d"
import { InteractionHijacker, InteractionHijackerDelegate } from "./label2d"

export class Polygon2DBoundaryCloner extends InteractionHijacker {
  private _target: Polygon2D
  private _initialPoints: PathPoint2D[]
  private _label: Polygon2D | undefined
  private _handler1Idx: number | undefined
  private _handler2Idx: number | undefined
  private _reversed: boolean

  constructor (target: Label2D, delegate: InteractionHijackerDelegate) {
    super(delegate)
    this._target = target as Polygon2D
    this._initialPoints = [...this._target.points]
    this._reversed = false
  }

  public onClickHandler(label: Label2D, handlerIdx: number) {
    if (this._label === undefined || this._handler1Idx === undefined || this._label.labelId !== label.labelId) {
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

  public onKeyDown(e: KeyboardEvent) {
    switch (e.key) {
      case Key.ALT:
        this._reversed = !this._reversed
        this.updateRender()
        break
      case Key.ENTER:
        this._delegate.didFinish(this)
        break
    }
  }

  private updateRender(): void {
    const { _label: source, _reversed: reversed, _handler1Idx: h1, _handler2Idx: h2 } = this
    if (source === undefined || h1 === undefined || h2 === undefined) {
      return
    }

    const qs = source.points
    const ps = [...this._initialPoints]
    const advance = reversed ? (i: number) => (i - 2 + qs.length) % qs.length + 1 : (i: number) => i % qs.length + 1
    for (let idx = h1; idx !== h2; idx = advance(idx)) {
      const q = qs[idx - 1]
      ps.push(q)
    }
    const q = qs[h2 - 1].clone()
    ps.push(q)

    this._target.points = ps
  }
}