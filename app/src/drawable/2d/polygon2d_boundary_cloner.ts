import Session from "../../common/session"
import { updatePolygon2DBoundaryCloneStatus } from "../../action/common"
import { alert } from "../../common/alert"
import { Severity } from "../../types/common"

import { Key, LabelTypeName } from "../../const/common"
import { Label2D, Label2DModifier } from "./label2d"
import { Polygon2D } from "./polygon2d"
import { PathPoint2D } from "./path_point2d"
import { PathPointType } from "../../types/state"

/**
 * Polygon2DBoundaryCloner modifies a single Pylogon2D by cloning a segment
 * of the boundary of some other Polygon2D to the target one.
 */
export class Polygon2DBoundaryCloner extends Label2DModifier {
  private _target: Polygon2D
  private readonly _initialPoints: PathPoint2D[]

  private _label: Polygon2D | undefined
  private _handler1Idx: number | undefined
  private _handler2Idx: number | undefined
  private _reverse: boolean

  private _finishCallback: (() => void) | undefined

  /**
   * Constructor
   *
   * @param target the label to append the cloned boundary segment.
   */
  constructor(target: Polygon2D) {
    super()

    this._target = target
    this._initialPoints = [...this._target.points]
    this._reverse = false

    this.syncState()
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
      alert(Severity.WARNING, `Can select only normal vertex`)
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
      this._label = l
      this._handler1Idx = handlerIdx
      this._handler2Idx = undefined
      this._reverse = false
      this.reset()
    } else {
      // Set current handler to be the second vertex
      if (this._handler1Idx === handlerIdx) {
        // Must select a diffent one
        alert(Severity.WARNING, `The second handler must be a different one.`)
        return
      }

      this._handler2Idx = handlerIdx
      this.update()
    }

    this.syncState()
  }

  /**
   * Implementation of the `onKeyDown` abstract method.
   *
   * @param e
   */
  public onKeyDown(e: KeyboardEvent): void {
    switch (e.key) {
      case Key.ALT:
        this._reverse = !this._reverse
        this.update()
        this.syncState()
        break
      case Key.ENTER:
        this.finish()
        break
      case Key.ESCAPE:
        this.reset()
        this.finish()
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
   * Reset to initial points.
   */
  private reset(): void {
    this._target.points = [...this._initialPoints]
  }

  /**
   * Update the rendering of the target for preview purpose
   */
  private update(): void {
    const {
      _label: source,
      _handler1Idx: h1,
      _handler2Idx: h2,
      _reverse: reverse
    } = this
    if (source === undefined || h1 === undefined || h2 === undefined) {
      return
    }

    const qs = source.points

    const n = qs.length // should be >= 2 upon reaching here
    const advance = reverse
      ? (i: number) => ((i - 2 + n) % n) + 1
      : (i: number) => (i % n) + 1

    const ps = [...this._initialPoints]
    for (let idx = h1; idx !== h2; idx = advance(idx)) {
      const q = qs[idx - 1]
      ps.push(q)
    }
    const q = qs[h2 - 1].clone()
    ps.push(q)

    this._target.points = ps
  }

  /**
   * Finished.
   */
  private finish(): void {
    this._finishCallback?.()
    Session.dispatch(updatePolygon2DBoundaryCloneStatus(undefined))
  }

  /**
   * Sync the update to the global state.
   */
  private syncState(): void {
    Session.dispatch(
      updatePolygon2DBoundaryCloneStatus({
        labelId: this._label?.labelId,
        handler1Idx: this._handler1Idx,
        handler2Idx: this._handler2Idx,
        reverse: this._reverse
      })
    )
  }
}
