import HttpException from './httpException'

/**
 * Class represents server side unknown error
 */
class UserExistsException extends HttpException {
  /**
   * Create a server error exception
   */
  constructor () {
    super(400, 'User with the email already exist')
  }
}

export default UserExistsException
