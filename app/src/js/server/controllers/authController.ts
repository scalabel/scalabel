import { compare, hash } from 'bcrypt'
import { NextFunction, Request, Response, Router } from 'express'
import { sign } from 'jsonwebtoken'

import DataStoredInToken from '../dto/dataStoredInToken'
import ForgetPasswordDto from '../dto/forgetPassword'
import LogInDto from '../dto/login'
import ResetPasswordDto from '../dto/resetPassword'
import TokenData from '../dto/tokenData'
import { resetModel } from '../entities/reset'
import { User, userModel } from '../entities/user'
import { NoSuchUserException, ServerErrorException, WrongCredentialsException } from '../exceptions'
import validationMiddleware from '../middlewares/validation'
import { API_PATH } from '../path'
import { ServerConfig } from '../types'

/** Authentication controller for login and reset */
class AuthController {

  /** Root path for this controller */
  public path = '/auth'
  /** Server config */
  private config: ServerConfig
  /** Router object */
  private router = Router()
  /** User repo */
  private userModel = userModel
  /** Reset repo */
  private resetModel = resetModel

  /**
   * Create an authentication controller
   * @param {ServerConfig} config - Server config
   */
  constructor (config: ServerConfig) {
    this.config = config
    this.initialRoutes()
  }

  /**
   * Login function
   * @param {Request} request - request object
   * @param {Response} response - response object
   * @param {NextFunction} next - function
   */
  public async login (
    request: Request, response: Response, next: NextFunction) {
    const logInData: LogInDto = request.body
    this.userModel.findOne({ email: logInData.email }, async (err, user) => {
      if (err) {
        next(new WrongCredentialsException())
      } else {
        if (user) {
          const isPasswordMatching = await compare(
            logInData.password,
            user.get('password', null, { getters: false })
          )
          if (isPasswordMatching) {
            const tokenData = this.createToken(user)
            response.setHeader('Set-Cookie', [this.createCookie(tokenData)])
            response.send(user)
          } else {
            next(new WrongCredentialsException())
          }
        } else {
          next(new WrongCredentialsException())
        }
      }
    })
  }

  /**
   * Reset password function
   * @param {Request} request - request object
   * @param {Response} response - response object
   * @param {NextFunction} next - next function
   */
  public resetPassword (
    request: Request, response: Response, next: NextFunction) {
    const resetPasswordDto: ResetPasswordDto = request.body
    this.resetModel.findOne({ token: resetPasswordDto.token }).then((reset) => {
      if (reset) {
        this.userModel.findById(reset.userId).then(async (user) => {
          if (user) {
            const hashedPassword = await hash(resetPasswordDto.password, 10)
            user.password = hashedPassword
            return user.save()
          } else {
            next(new WrongCredentialsException())
          }
        }).catch(() => next(new WrongCredentialsException()))
      } else {
        next(new WrongCredentialsException())
      }
    }).then(() => response.send(200))
    .catch(() => next(new WrongCredentialsException()))
  }

  /**
   * forget password function
   * @param {Request} request - request object
   * @param {Response} response - response object
   * @param {NextFunction} next - next function
   */
  public forgetPassword (
    request: Request, response: Response, next: NextFunction) {
    const forgetPasswordDto: ForgetPasswordDto = request.body
    this.userModel.findOne({ email: forgetPasswordDto.email },
      async (err, user) => {
        if (err) {
          next(new ServerErrorException())
        } else {
          if (user) {
            // TODO: Send email
            response.send(200)
          } else {
            next(new NoSuchUserException())
          }
        }
      }
    )
  }

  /**
   * Logout function
   * @param {Request} request - Request object
   * @param {Response} response - Response object
   */
  public logout (_request: Request, response: Response) {
    response.setHeader('Set-Cookie', ['Authorization=;Max-age=0'])
    response.send(200)
  }

  /**
   * Initial base routes for this controller
   */
  private initialRoutes () {
    this.router.post(
      `${API_PATH}${this.path}/login`,
      validationMiddleware(LogInDto),
      this.login)
    this.router.post(
      `${API_PATH}${this.path}/forget_password`,
      validationMiddleware(ForgetPasswordDto),
      this.forgetPassword)
    this.router.post(
      `${API_PATH}${this.path}/reset_password`,
      validationMiddleware(ResetPasswordDto),
      this.resetPassword)
    this.router.post(
      `${API_PATH}${this.path}/logout`,
      this.logout)
  }

  /**
   * Create cookie
   * @param {TokenData} tokenData - Token data
   * @returns {string} auth cookie
   */
  private createCookie (tokenData: TokenData): string {
    return `Authorization=${tokenData.token}; HttpOnly; Max-Age=${tokenData.expiresIn}`
  }

  /**
   * Create Token
   * @param {User} user - user info
   * @returns {TokenData} token info
   */
  private createToken (user: User): TokenData {
    const expiresIn = 60 * 60 // an hour
    const secret = this.config.jwtSecret
    const dataStoredInToken: DataStoredInToken = {
      _id: user._id
    }
    return {
      expiresIn,
      token: sign(dataStoredInToken, secret, { expiresIn })
    }
  }
}

export default AuthController
