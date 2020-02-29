import { State } from '../functional/types'
import { RegisterMessageType, SyncActionMessageType } from './types'

type socketHandlerType =
  (() => {}) |
  ((data: SyncActionMessageType) => {}) |
  ((data: RegisterMessageType) => {})

type emitDataType = SyncActionMessageType | State

export type redisHandlerType =
  ((channel: string, value: string) => {})

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

/**
 * Generic interface for key value client
 * Redis implements this
 */
export interface KeyValueClient {
  /** Add a handler function */
  on (event: string, callback: redisHandlerType): void

  /** Subscribe to a channel */
  subscribe (channel: string): void

  /** Delete a value */
  del (key: string): void

  /** Start an atomic transaction */
  multi (): RedisMulti

  /** Get a value */
  get (key: string): Promise<string>

  /** Set value with timeout */
  psetex (key: string, timeout: number, val: string): Promise<void>

  /** Increment a value */
  incr (key: string): Promise<void>

  /** Modify the config */
  config (type: string, name: string, value: string): void
}
