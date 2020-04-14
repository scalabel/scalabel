import { compare, hash } from 'bcrypt'
import crypto from 'crypto'
import { sign } from 'jsonwebtoken'
import CreateUserDto from '../dto/createUser'
import DataStoredInToken from '../dto/dataStoredInToken'
import ForgetPasswordDto from '../dto/forgetPassword'
import LogInDto from '../dto/login'
import ResetPasswordDto from '../dto/resetPassword'
import TokenData from '../dto/tokenData'
import { Reset, resetModel } from '../entities/reset'
import { User, userModel } from '../entities/user'
import { NoSuchUserException, ResetTokenExpiredException, ServerErrorException, UserExistsException, WrongCredentialsException } from '../exceptions'
import { ServerConfig } from '../types'
import { AuthenticationImpl } from './auth'

/**
 * Authentication service
 */
class AuthenticationLocalService implements AuthenticationImpl {

  /** User repo */
  private userModel = userModel
  /** Reset repo */
  private resetModel = resetModel
  /** Config */
  private config: ServerConfig

  /**
   * Create an authentication controller
   * @param {ServerConfig} config - Server config
   */
  constructor (config: ServerConfig) {
    this.config = config
    if (!config.jwtSecret) {
      throw new Error('Local auth config error')
    }
  }

  /**
   * Create user
   * @param {CreateUserDto} registerData - Create user dto
   * @param {AuthenticationImpl~registerCallback} success - Handle success
   * @param {AuthenticationImpl~errorCallback} error - Handle error
   */
  public register = (registerData: CreateUserDto,
                     success: (user: User) => void,
                     error: (error: Error) => void) => {
    this.userModel.findOne({ email: registerData.email }, (err, user) => {
      if (err) {
        error(new ServerErrorException())
      } else if (user) {
        error(new UserExistsException())
      } else {
        hash(registerData.password, 10).then((hashedPassword) => {
          this.userModel.create({
            ...registerData,
            password: hashedPassword
          }).then((newUser) => success(newUser))
          .catch(() => error(new ServerErrorException()))
        }).catch(() => error(new ServerErrorException()))
      }
    })
  }

  /**
   * Login
   * @param logInData login data
   * @param {AuthenticationImpl~loginCallback} success - Handle success
   * @param {AuthenticationImpl~errorCallback} error - Handle error
   */
  public login = (logInData: LogInDto,
                  success: (user: User) => void,
                  error: (error: Error) => void) => {
    this.userModel.findOne({ email: logInData.email }, async (err, user) => {
      if (err) {
        error(new WrongCredentialsException())
      } else {
        if (user) {
          const isPasswordMatching = await compare(
            logInData.password,
            user.get('password', null, { getters: false })
          )
          if (isPasswordMatching) {
            success(user)
          } else {
            error(new WrongCredentialsException())
          }
        } else {
          error(new WrongCredentialsException())
        }
      }
    })
  }

  /**
   * Reset password
   * @param {ResetPasswordDto} resetData - Reset password data
   * @param {AuthenticationImpl~resetCallback} success - Handle success
   * @param {AuthenticationImpl~errorCallback} error - Handle error
   */
  public resetPassword = (resetData: ResetPasswordDto,
                          success: () => void,
                          error: (error: Error) => void) => {
    this.resetModel.findOne({ token: resetData.token }).then((reset) => {
      if (reset) {
        if (reset.applied ||
          new Date().getTime() > reset.createdAt.getTime() + 15 * 60 * 1000) {
          error(new ResetTokenExpiredException())
        }
        this.userModel.findById(reset.userId).then(async (user) => {
          if (user) {
            const hashedPassword = await hash(resetData.password, 10)
            user.password = hashedPassword
            reset.applied = true
            reset.save().catch()
            return user.save()
          } else {
            error(new ServerErrorException())
          }
        }).catch(() => error(new ServerErrorException()))
      } else {
        error(new WrongCredentialsException())
      }
    }).then(() => success())
  .catch(() => error(new WrongCredentialsException()))
  }

  /**
   * Forget password logic handling
   * @param {ForgetPasswordDto} forgetPasswordDto - forget password data
   * @param {AuthenticationImpl~forgetCallback} success - Handle success
   * @param {AuthenticationImpl~errorCallback} error - Handle error
   */
  public forgetPassword = (forgetPasswordDto: ForgetPasswordDto,
                           success: (reset: Reset) => void,
                           error: (error: Error) => void) => {
    this.userModel.findOne({ email: forgetPasswordDto.email },
        async (err, user) => {
          if (err) {
            error(new ServerErrorException())
          } else {
            if (user) {
              // Call email service to send a reset email.
              const reset = await this.resetModel.create({
                userId: user._id,
                token: crypto.randomBytes(64).toString('hex'),
                applied: false,
                createdAt: new Date()
              })
              success(reset)
            } else {
              error(new NoSuchUserException())
            }
          }
        }
      )
  }

  /**
   * Get cookie
   * @param {TokenData} tokenData - Token data
   * @returns {string} auth cookie
   */
  public getCookie (tokenData: TokenData): string {
    return `Authorization=${tokenData.token}; HttpOnly; Max-Age=${tokenData.expiresIn}`
  }

  /**
   * Get Token
   * @param {User} user - user info
   * @returns {TokenData} token info
   */
  public getToken = (user: User): TokenData => {
    const expiresIn = 60 * 60 // an hour
    const secret = this.config.jwtSecret
    if (!secret) {
      throw new Error('JWT secret missed')
    }
    const dataStoredInToken: DataStoredInToken = {
      _id: user._id
    }
    return {
      expiresIn,
      token: sign(dataStoredInToken, secret, { expiresIn })
    }
  }
}

export default AuthenticationLocalService
