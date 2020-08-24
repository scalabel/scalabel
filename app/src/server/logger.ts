import winston from 'winston'
import { MaybeError } from '../types/common'
import { hostname, now } from './path'

/**
 * Logger for console messages
 */
class Logger {
  /** use winston to manage the actual prints */
  private _logger: winston.Logger
  /** whether to mute the info logging */
  private _silent: boolean
  /** log transports */
  private transport: winston.transport

  constructor () {
    this.transport = new winston.transports.Console({ level: 'info' })

    this._logger = winston.createLogger({
      format: winston.format.combine(
        winston.format.timestamp({
          format: now
        }),
        winston.format.printf(({ level, message, label, timestamp }) => {
          let labelString = ''
          if (label !== undefined) {
            labelString = `[${label}] `
          }
          return `${hostname()}:${process.pid}:${timestamp} ${labelString}${level}: ${message}`
        })
      ),
      transports: [
        this.transport
      ],
      exitOnError: true
    })
    this._silent = false
  }

  /** set log verbose level */
  public setLogLevel (level: string): void {
    this.transport.level = level
  }

  /** print errors */
  public error (err: MaybeError): void {
    if (err) {
      this._logger.log({
        level: 'error',
        message: err.message,
        trace: err.stack
      })
    }
  }

  /** print informative messages */
  public info (message: string): void {
    if (message && !this._silent) {
      this._logger.log({
        level: 'info',
        message
      })
    }
  }

    /** print informative messages */
  public debug (message: string): void {
    if (message && !this._silent) {
      this._logger.log({
        level: 'debug',
        message
      })
    }
  }

  /**
   * whether to mute info logging.
   * It can provide a clean console for unit test
   */
  public mute (silent: boolean = true): void {
    this._silent = silent
  }
}

export default new Logger()
