import { Request, Response, Router } from "express"
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

  /**
   * Constructor
   *
   * @param config
   */
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
   */
  private readonly decode = (request: Request, response: Response): void => {
    if (this.config.user.on && this.config.cognito !== undefined) {
      const cognito = this.config.cognito
      this.exchangeCode(request.query.code as string)
        .then((tokens: TokenSet) => {
          response.render("callback", {
            accessToken: tokens.access_token,
            idToken: tokens.id_token,
            refreshToken: tokens.refresh_token,
            tokenType: tokens.token_type,
            uri: cognito.userPoolBaseUri,
            clientId: cognito.clientId
          })
        })
        .catch(() => {})
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
   */
  private readonly exchangeCode = async (code: string): Promise<TokenSet> => {
    if (this.config.cognito === undefined) {
      throw new Error("cognito is not configured")
    }
    const cognito = this.config.cognito
    return await new Promise((resolve, reject) => {
      const postData = querystring.stringify({
        grant_type: "authorization_code",
        client_id: cognito.clientId,
        redirect_uri: cognito.callbackUri,
        code
      })
      const config = {
        host: cognito.userPoolBaseUri,
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
  private readonly initialRoutes = (): void => {
    this.router.get("/", this.decode)
  }
}

export default Callback
