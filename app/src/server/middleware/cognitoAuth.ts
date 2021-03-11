import { NextFunction, Request, Response } from "express"
import jwt from "express-jwt"
import jwksRsa from "jwks-rsa"

import { ServerConfig } from "../../types/config"

const auth = (
  config: ServerConfig
): ((request: Request, response: Response, n: NextFunction) => void) => {
  if (config.cognito !== undefined) {
    return jwt({
      secret: jwksRsa.expressJwtSecret({
        cache: true,
        rateLimit: true,
        jwksRequestsPerMinute: 5,
        jwksUri: `https://cognito-idp.${config.cognito.region}.amazonaws.com/${config.cognito.userPool}/.well-known/jwks.json`
      }),

      issuer: `https://cognito-idp.${config.cognito.region}.amazonaws.com/${config.cognito.userPool}`,
      algorithms: ["RS256"]
    })
  } else {
    return (_request: Request, _response: Response, next: NextFunction) =>
      next()
  }
}

export default auth
