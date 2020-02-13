import { State } from '../functional/types'
import { ActionPacketType } from '../server/types'

type HandlerFnType =
  (() => void) |
  ((state: State) => void) |
  ((actions: ActionPacketType) => void)

export interface ClientSocket {
  /** whether connection is established */
  connected: boolean
  /** subscribe a handler */
  on (eventName: string, handler: HandlerFnType): void
  /** send a message */
  emit (eventName: string, data: string): void
}

/** A dummy client socket for testing */
export class DummySocket implements ClientSocket {
  /** whether connection is established */
  public connected: boolean

  constructor () {
    this.connected = true
  }

  /**
   * A dummy method for registering handlers
   */
  public on (_eventName: string, _handler: () => void) {
    return
  }

  /**
   * A dummy method for emitting messages
   */
  public emit (_eventName: string, _data: string) {
    return
  }
}
