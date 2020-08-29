import { NextFunction, Request, Response, Router } from "express"
import https from "https"
import querystring from "querystring"
import { ServerConfig } from "../../types/config"
import errorHandler from "../middleware/errorHandler"

/**
 * Token set for exchange
 *
 * @interface TokenSet
 */
interface TokenSet {
  /**
   * ID token
   *
   * @type {string}
   * @memberof TokenSet
   */
  id_token: string
  /**
   * Access Token
   *
   * @type {string}
   * @memberof TokenSet
   */
  access_token: string
  /**
   * Refresh Token
   *
   * @type {string}
   * @memberof TokenSet
   */
  refresh_token: string
  /**
   * Expires time
   *
   * @type {number}
   * @memberof TokenSet
   */
  expires: number
  /**
   * Token type
   *
   * @type {string}
   * @memberof TokenSet
   */
  token_type: string
}

/**
 * Authenticate callback
 *
 * @class Callback
 */
class Callback {
  /**
   * Express router
   *
   * @type {Router}
   * @memberof Callback
   */
  public router: Router
  /**
   * Server config
   *
   * @private
   * @type {ServerConfig}
   * @memberof Callback
   */
  private readonly config: ServerConfig

  constructor(config: ServerConfig) {
    this.config = config
    this.router = Router()
    this.initialRoutes()
    this.router.use(errorHandler)
  }

  /**
   * decode
   *
   * @private
   * @param {Request} request - Request
   * @param {Response} response - Response
   * @param {NextFunction} next - Next Function
   * @memberof Callback
   */
  private readonly decode = (
    request: Request,
    response: Response,
    _next: NextFunction
  ) => {
    if (this.config.user.on && this.config.cognito) {
      this.exchangeCode(request.query.code as string)
        .then((tokens: TokenSet) => {
          response.render("callback", {
            accessToken: tokens.access_token,
            idToken: tokens.id_token,
            refreshToken: tokens.refresh_token,
            tokenType: tokens.token_type,
            uri: this.config.cognito!.userPoolBaseUri,
            clientId: this.config.cognito!.clientId
          })
        })
        .catch()
    } else {
      response.send(200)
    }
  }

  /**
   * Exchange cognito for tokens
   *
   * @private
   * @param {string} code
   * @returns {Promise<JSON>}
   * @memberof Callback
   */
  private readonly exchangeCode = (code: string): Promise<TokenSet> => {
    return new Promise((resolve, reject) => {
      const postData = querystring.stringify({
        grant_type: "authorization_code",
        client_id: this.config.cognito!.clientId,
        redirect_uri: this.config.cognito!.callbackUri,
        code
      })
      const config = {
        host: this.config.cognito!.userPoolBaseUri,
        path: "/oauth2/token",
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded"
        }
      }
      const request = https.request(config, (response) => {
        if (response.statusCode === 200) {
          response.on("data", (data) => {
            const jsonString = data.toString("UTF-8")
            const jsonData = JSON.parse(jsonString) as TokenSet
            resolve(jsonData)
          })
        } else {
          reject(new Error("Server exception"))
        }
      })
      request.on("error", () => {
        reject(new Error("Server exception"))
      })
      request.write(postData)
      request.end()
    })
  }

  /**
   * Initial routes for the controller
   */
  private readonly initialRoutes = () => {
    this.router.get("/", this.decode)
  }
}

export default Callback
