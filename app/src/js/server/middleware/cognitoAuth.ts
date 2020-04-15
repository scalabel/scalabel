import jwt from 'express-jwt'
import jwksRsa from 'jwks-rsa'

import { ServerConfig } from '../types'

const auth = (config: ServerConfig) =>
  jwt({
    secret: jwksRsa.expressJwtSecret({
      cache: true,
      rateLimit: true,
      jwksRequestsPerMinute: 5,
      jwksUri: `https://cognito-idp.${config.region}.amazonaws.com/${config.userPool}/.well-known/jwks.json`
    }),

    issuer: `https://cognito-idp.${config.region}.amazonaws.com/${config.userPool}`,
    algorithms: [ 'RS256' ]
  })

export default auth
