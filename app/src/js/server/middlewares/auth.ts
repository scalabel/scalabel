
import { NextFunction, Response } from 'express'
import { verify } from 'jsonwebtoken'
import DataStoredInToken from '../dto/dataStoredInToken'
import RequestWithUser from '../dto/requestWithUser'
import { userModel } from '../entities/user'
import { AuthenticationTokenMissingException, ServerErrorException, WrongAuthenticationTokenException } from '../exceptions'
import { ServerConfig } from '../types'

export const checkToken = (config: ServerConfig) => {
  return (request: RequestWithUser,
          _response: Response,
          next: NextFunction) => {
    const cookies = request.cookies
    if (cookies && cookies.Authorization) {
      const secret = config.jwtSecret || ''
      try {
        const verificationResponse =
          verify(cookies.Authorization, secret) as DataStoredInToken
        const id = verificationResponse._id
        userModel.findById(id, (err, user) => {
          if (err) {
            next(new ServerErrorException())
          } else {
            if (user) {
              request.user = user
              next()
            } else {
              next(new WrongAuthenticationTokenException())
            }
          }
        })
      } catch (error) {
        next(new WrongAuthenticationTokenException())
      }
    } else {
      next(new AuthenticationTokenMissingException())
    }
  }
}
