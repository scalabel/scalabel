import { State } from '../functional/types'
import { RegisterMessageType, SyncActionMessageType } from './types'

type socketHandlerType =
  (() => {}) |
  ((data: SyncActionMessageType) => {}) |
  ((data: RegisterMessageType) => {})

type emitDataType = SyncActionMessageType | State

export type redisHandlerType =
  ((channel: string, value: string) => {}) |
  ((channel: string, value: string) => void)

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

/** Redis multi object for atomic transactions */
export interface RedisMulti {
  /** Add a command to the transaction */
  psetex (key: string, timeout: number, val: string): void

  /** Execute the transaction */
  exec (): void
}
