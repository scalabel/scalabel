import { Request, Response } from "express"

import { ServerConfig } from "../../types/config"
import { HttpException } from "../exception"

/**
 * Error handler
 *
 * @param {Error} err - error
 * @param {Request} req - request object
 * @param {Response} res - response object
 * @param {NextFunction} next - next function
 * @param config
 */
const errorHandler = (config: ServerConfig) => (
  error: HttpException,
  _request: Request,
  response: Response
) => {
  const status = error.status !== 0 ? error.status : 500
  const message = error.message !== "" ? error.message : "Something went wrong"
  const resData: { [k: string]: string } = {
    code: status.toString(),
    data: message
  }
  if (status === 401 && config.user.on && config.cognito !== undefined) {
    resData.redirect = `https://${config.cognito.userPoolBaseUri}/login?client_id=${config.cognito.clientId}&response_type=code&redirect_uri=${config.cognito.callbackUri}`
  }
  response.status(status).json(resData)
}

export default errorHandler
