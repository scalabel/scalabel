import winston from 'winston'
import { MaybeError } from './types'

/**
 * Logger for console messages
 */
class Logger {
  /** use winston to manage the actual prints */
  private _logger: winston.Logger

  constructor () {
    this._logger = winston.createLogger({
      transports: [
        new winston.transports.Console()
      ],
      exitOnError: false
    })
  }

  /** print errors */
  public error (err: MaybeError): void {
    if (err) {
      this._logger.log({
        level: 'error',
        message: err.message
      })
    }
  }

  /** print informative messages */
  public info (message: string): void {
    if (message) {
      this._logger.log({
        level: 'info',
        message
      })
    }
  }
}

export default new Logger()
