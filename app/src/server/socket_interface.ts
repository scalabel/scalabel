import { RegisterMessageType, SyncActionMessageType } from "../types/message"
import { State } from "../types/state"

type socketHandlerType =
  | (() => {})
  | ((data: SyncActionMessageType) => {})
  | ((data: RegisterMessageType) => {})

type serverEmitType = SyncActionMessageType | State

/**
 * Interface for subset of socket.io functionality used by the server
 */
export interface SocketServer {
  /** Socket id */
  id: string

  /** Message broadcaster */
  broadcast: {
    /** Object for specifying target */
    to: (
      room: string
    ) => {
      /** Message echoer */
      emit: (event: string, data: serverEmitType) => void
    }
  }

  /** Subscribe a handler */
  join: (room: string) => void

  /** Echo a message */
  emit: (event: string, data: serverEmitType) => void

  /** Add a handler function */
  on: (event: string, callback: socketHandlerType) => void
}

type clientEmitType = SyncActionMessageType | RegisterMessageType

/**
 * Interface for subset of socket.io functionality used by the frontend
 */
export interface SocketClient {
  /** Connection status */
  connected: boolean

  /** Message sending */
  emit: (event: string, data: clientEmitType) => void
}
