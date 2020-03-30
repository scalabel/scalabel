import { IsEmail } from 'class-validator'

/**
 * Represents forget password dto
 */
class ForgetPasswordDto {
  /**
   * email
   */
  @IsEmail()
  public email: string = ''
}

export default ForgetPasswordDto
