import winston from "winston"

import { MaybeError } from "../types/common"
import { hostname, now } from "./path"

/**
 * Logger for console messages
 */
class Logger {
  /** use winston to manage the actual prints */
  private readonly _logger: winston.Logger
  /** whether to mute the info logging */
  private _silent: boolean
  /** log transports */
  private transport: winston.transport

  /**
   * Constructor
   */
  constructor() {
    this.transport = new winston.transports.Console({ level: "info" })

    this._logger = winston.createLogger({
      format: winston.format.combine(
        winston.format.timestamp({
          format: now
        }),
        winston.format.printf(({ level, message, label, timestamp }) => {
          let labelString = ""
          if (label !== undefined) {
            // eslint-disable-next-line @typescript-eslint/restrict-template-expressions
            labelString = `[${label}] `
          }
          return `${hostname()}:${
            process.pid
            // eslint-disable-next-line @typescript-eslint/restrict-template-expressions
          }:${timestamp} ${labelString}${level}: ${message}`
        })
      ),
      transports: [this.transport],
      exitOnError: true
    })
    this._silent = false
  }

  /**
   * set log verbose level
   *
   * @param level
   */
  public setLogLevel(level: string): void {
    this.transport.level = level
  }

  /**
   * print errors
   *
   * @param err
   */
  public error(err: MaybeError): void {
    if (err !== undefined) {
      this._logger.log({
        level: "error",
        message: err.message,
        trace: err.stack
      })
    }
  }

  /**
   * print informative messages
   *
   * @param message
   */
  public info(message: string): void {
    if (message !== "" && !this._silent) {
      this._logger.log({
        level: "info",
        message
      })
    }
  }

  /**
   * print informative messages
   *
   * @param message
   */
  public debug(message: string): void {
    if (message !== "" && !this._silent) {
      this._logger.log({
        level: "debug",
        message
      })
    }
  }

  /**
   * whether to mute info logging.
   * It can provide a clean console for unit test
   *
   * @param silent
   */
  public mute(silent: boolean = true): void {
    this._silent = silent
  }
}

export default new Logger()
