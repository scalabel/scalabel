import { NextFunction, Request, Response } from 'express'
import { HttpException } from '../exception'

/**
 * Error handler
 * @param {Error} err - error
 * @param {Request} req - request object
 * @param {Response} res - response object
 * @param {NextFunction} next - next function
 */
const errorHandler = (
  error: HttpException,
  _request: Request,
  response: Response,
  _next: NextFunction
) => {
  const status = error.status || 500
  const message = error.message || 'Something went wrong'
  response.status(status).json({
    code: status,
    data: message
  })
}

export default errorHandler
