import { State } from '../functional/types'
import { ActionPacketType } from './types'

type onHandlerType = (() => {}) | ((data: string) => {})

type emitDataType = ActionPacketType | State

export interface SocketServer {
  /** socket id */
  id: string

  /** message broadcaster */
  broadcast: {
    /** object for specifying target */
    to: (room: string) => {
      /** message echoer */
      emit: (event: string, data: emitDataType) => void
    }
  }

  /** subscribe a handler */
  join (room: string): void

  /** echo a message */
  emit (event: string, data: emitDataType): void

  /** add a handler funciton */
  on (event: string, callback: onHandlerType): void
}
