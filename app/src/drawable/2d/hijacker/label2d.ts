import { Label2D } from "../label2d"

export abstract class InteractionHijacker {
  protected _delegate: InteractionHijackerDelegate

  constructor(delegate: InteractionHijackerDelegate) {
    this._delegate = delegate
  }

  public abstract onClickHandler(label: Label2D, handlerIdx: number): void
  public abstract onKeyDown(e: KeyboardEvent): void
}

export abstract class InteractionHijackerDelegate {
  public abstract didFinish(hijacker: InteractionHijacker): void
}
