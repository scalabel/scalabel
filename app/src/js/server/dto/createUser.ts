import { IsEmail, IsNotEmpty, IsString } from 'class-validator'

/**
 * Create user
 */
class CreateUserDto {
  /** First name */
  @IsString()
  public firstName: string = ''

  /** Last name */
  @IsString()
  public lastName: string = ''

  /** Email */
  @IsEmail()
  public email: string = ''

  /** Password */
  @IsString()
  @IsNotEmpty()
  public password: string = ''
}

export default CreateUserDto
