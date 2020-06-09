import { State } from '../functional/types'
import { RegisterMessageType, SyncActionMessageType } from './types'

type socketHandlerType =
  (() => {}) |
  ((data: SyncActionMessageType) => {}) |
  ((data: RegisterMessageType) => {})

type emitDataType = SyncActionMessageType | State

/**
 * Generic interface for server-side socket
 * Socket.io implements this
 */
export interface SocketServer {
  /** Socket id */
  id: string

  /** Message broadcaster */
  broadcast: {
    /** Object for specifying target */
    to: (room: string) => {
      /** Message echoer */
      emit: (event: string, data: emitDataType) => void
    }
  }

  /** Subscribe a handler */
  join (room: string): void

  /** Echo a message */
  emit (event: string, data: emitDataType): void

  /** Add a handler function */
  on (event: string, callback: socketHandlerType): void
}

type clientEmitType = SyncActionMessageType | RegisterMessageType

/**
 * Generic interface for frontend socket
 * Socket.io implements this
 */
export interface SocketClient {
  /** Connection status */
  connected: boolean

  /** Message sending */
  emit (event: string, data: clientEmitType): void
}
