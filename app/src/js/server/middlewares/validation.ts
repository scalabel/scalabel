import { plainToClass } from 'class-transformer'
import { ClassType } from 'class-transformer/ClassTransformer'
import { validate, ValidationError } from 'class-validator'
import { RequestHandler } from 'express'
import { HttpException, ParameterException } from '../exceptions'

/**
 * Validate request parameters
 * @param {ClassType} type - Request object class
 * @param {boolean} skipMissingProperties - Where skip missing properties
 */
function validationMiddleware<T> (
  type: ClassType<T>,
  skipMissingProperties: boolean = false): RequestHandler {
  return (req, _res, next) => {
    if (req.body.constructor === Object && Object.keys(req.body).length === 0) {
      next(new ParameterException())
    }
    validate(plainToClass(type, req.body), { skipMissingProperties })
      .then((errors: ValidationError[]) => {
        if (errors.length > 0) {
          const message = errors.map(
            (error: ValidationError) => Object.values(error.constraints)).join(', ')
          next(new HttpException(400, message))
        } else {
          next()
        }
      }).catch()
  }
}

export default validationMiddleware
