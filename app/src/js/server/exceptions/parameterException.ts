import HttpException from './httpException'

/**
 * Class represents parameter error exception
 * @extends HttpException
 */
class ParameterException extends HttpException {
  /**
   * Create a parameter error exception
   */
  constructor () {
    super(400, 'Bad request')
  }
}

export default ParameterException
