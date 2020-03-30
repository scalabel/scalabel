import { Request } from 'express'
import { User } from '../entities/user'

/**
 * Represents request with user object
 */
interface RequestWithUser extends Request {
  /**
   * user
   */
  user: User
}

export default RequestWithUser
