import { NextFunction, Request, Response } from 'express'
import { HttpException } from '../exceptions'

/**
 * Error handler
 * @param {Error} err - error
 * @param {Request} req - request object
 * @param {Response} res - response object
 * @param {NextFunction} next - next function
 */
function errorHandler
  (error: HttpException, _request: Request,
   response: Response, _next: NextFunction) {
  const status = error.status || 500
  const message = error.message || 'Something went wrong'
  response
      .status(status)
      .json({
        message
      })
}

export default errorHandler
