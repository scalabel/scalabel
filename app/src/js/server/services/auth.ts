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

/**
 * Authentication service
 */
class AuthenticationService {

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
  }

  /**
   * Create user
   * @param {CreateUserDto} registerData - Create user dto
   */
  public register = async (registerData: CreateUserDto) => {
    if (await this.userModel.findOne({ email: registerData.email }).exec()) {
      throw new UserExistsException()
    }
    const hashedPassword = await hash(registerData.password, 10)
    const user = await this.userModel.create({
      ...registerData,
      password: hashedPassword
    })
    const tokenData = this.createToken(user)
    const cookie = this.createCookie(tokenData)
    return {
      cookie,
      user
    }
  }

  /**
   * Login
   * @param logInData login data
   */
  public login = (logInData: LogInDto) => {
    return new Promise<User>((resolve, reject) => {
      this.userModel.findOne({ email: logInData.email }, async (err, user) => {
        if (err) {
          reject(new WrongCredentialsException())
        } else {
          if (user) {
            const isPasswordMatching = await compare(
              logInData.password,
              user.get('password', null, { getters: false })
            )
            if (isPasswordMatching) {
              resolve(user)
            } else {
              reject(new WrongCredentialsException())
            }
          } else {
            reject(new WrongCredentialsException())
          }
        }
      })
    })
  }

  /**
   * Reset password
   * @param {ResetPasswordDto} resetData - Reset password data
   */
  public resetPassword = (resetData: ResetPasswordDto) => {
    return new Promise<number>((resolve, reject) => {
      this.resetModel.findOne({ token: resetData.token }).then((reset) => {
        if (reset) {
          if (reset.applied ||
            new Date().getTime() > reset.createdAt.getTime() + 15 * 60 * 1000) {
            reject(new ResetTokenExpiredException())
          }
          this.userModel.findById(reset.userId).then(async (user) => {
            if (user) {
              const hashedPassword = await hash(resetData.password, 10)
              user.password = hashedPassword
              reset.applied = true
              reset.save().catch()
              return user.save()
            } else {
              reject(new ServerErrorException())
            }
          }).catch(() => reject(new ServerErrorException()))
        } else {
          reject(new WrongCredentialsException())
        }
      }).then(() => resolve(200))
    .catch(() => reject(new WrongCredentialsException()))
    })
  }

  /**
   * Forget password logic handling
   * @param {ForgetPasswordDto} forgetPasswordDto - forget password data
   */
  public forgetPassword = (forgetPasswordDto: ForgetPasswordDto) => {
    return new Promise<Reset>((resolve, reject) => {
      this.userModel.findOne({ email: forgetPasswordDto.email },
        async (err, user) => {
          if (err) {
            reject(new ServerErrorException())
          } else {
            if (user) {
              // Call email service to send a reset email.
              const reset = await this.resetModel.create({
                userId: user._id,
                token: crypto.randomBytes(64).toString('hex'),
                applied: false,
                createdAt: new Date()
              })
              resolve(reset)
            } else {
              reject(new NoSuchUserException())
            }
          }
        }
      )
    })
  }

  /**
   * Create cookie
   * @param {TokenData} tokenData - Token data
   * @returns {string} auth cookie
   */
  public createCookie (tokenData: TokenData): string {
    return `Authorization=${tokenData.token}; HttpOnly; Max-Age=${tokenData.expiresIn}`
  }

  /**
   * Create Token
   * @param {User} user - user info
   * @returns {TokenData} token info
   */
  public createToken = (user: User): TokenData => {
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

export default AuthenticationService
