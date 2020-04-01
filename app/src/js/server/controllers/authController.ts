import { NextFunction, Request, Response, Router } from 'express'

import bodyParser from 'body-parser'
import CreateUserDto from '../dto/createUser'
import ForgetPasswordDto from '../dto/forgetPassword'
import LogInDto from '../dto/login'
import ResetPasswordDto from '../dto/resetPassword'
import errorHandler from '../middlewares/errorHandler'
import validationMiddleware from '../middlewares/validation'

import AuthenticationService from '../services/auth'
import { ServerConfig } from '../types'

/** Authentication controller for login and reset */
class AuthController {

  /** Root path for this controller */
  public path = '/auth'
  /** Router object */
  public router: Router
  /** Service */
  private service: AuthenticationService

  /**
   * Create an authentication controller
   * @param {ServerConfig} config - Server config
   */
  constructor (config: ServerConfig) {
    this.router = Router()
    this.router.use(bodyParser.json())
    this.service = new AuthenticationService(config)
    this.initialRoutes()
    this.router.use(errorHandler)
  }

  /**
   * Register user
   * @param {Request} request - request object
   * @param {Response} response - response object
   * @param {NextFunction} next - function
   */
  public registration =
    async (request: Request, response: Response, next: NextFunction) => {
      const userData: CreateUserDto = request.body
      try {
        const {
        cookie,
        user
      } = await this.service.register(userData)
        response.setHeader('Set-Cookie', [cookie])
        response.send(user)
      } catch (error) {
        next(error)
      }
    }

  /**
   * Login function
   * @param {Request} request - request object
   * @param {Response} response - response object
   * @param {NextFunction} next - function
   */
  public login = (request: Request, response: Response, next: NextFunction) => {
    const logInData: LogInDto = request.body
    this.service.login(logInData)
      .then((user) => {
        const tokenData = this.service.createToken(user)
        response.setHeader('Set-Cookie', [this.service.createCookie(tokenData)])
        response.send(user)
      }).catch((ex) => next(ex))
  }

  /**
   * Reset password function
   * @param {Request} request - request object
   * @param {Response} response - response object
   * @param {NextFunction} next - next function
   */
  public resetPassword =
    (request: Request, response: Response, next: NextFunction) => {
      const resetPasswordDto: ResetPasswordDto = request.body
      this.service.resetPassword(resetPasswordDto)
      .then((code) => response.send(code))
      .catch((ex) => next(ex))
    }

  /**
   * forget password function
   * @param {Request} request - request object
   * @param {Response} response - response object
   * @param {NextFunction} next - next function
   */
  public forgetPassword =
    (request: Request, response: Response, next: NextFunction) => {
      const forgetPasswordDto: ForgetPasswordDto = request.body
      this.service.forgetPassword(forgetPasswordDto)
      .then((code) => response.send(code))
      .catch((ex) => next(ex))
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
  private initialRoutes = () => {
    this.router.post(
      `${this.path}/register`,
      validationMiddleware(CreateUserDto),
      this.registration
    )
    this.router.post(
      `${this.path}/login`,
      validationMiddleware(LogInDto),
      this.login)
    this.router.post(
      `${this.path}/forget_password`,
      validationMiddleware(ForgetPasswordDto),
      this.forgetPassword)
    this.router.post(
      `${this.path}/reset_password`,
      validationMiddleware(ResetPasswordDto),
      this.resetPassword)
    this.router.post(
      `${this.path}/logout`,
      this.logout)
  }
}

export default AuthController
