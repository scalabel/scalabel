import { IsEmail, IsNotEmpty, IsString } from 'class-validator'

/**
 * Create user
 */
class CreateUserDto {
  /** Given name */
  @IsString()
  public givenName: string = ''

  /** Family name */
  @IsString()
  public familyName: string = ''

  /** Email */
  @IsEmail()
  public email: string = ''

  /** Password */
  @IsString()
  @IsNotEmpty()
  public password: string = ''
}

export default CreateUserDto
