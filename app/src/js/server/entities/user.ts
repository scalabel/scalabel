import * as mongoose from 'mongoose'

export interface User {
  /** Mongo id */
  _id: string
  /** First name */
  firstName: string
  /** Last name */
  lastName: string
  /** Full name */
  fullName: string
  /** Email */
  email: string
  /** Password */
  password: string
  /** Register time */
  createdAt: Date
}

const userSchema = new mongoose.Schema(
  {
    email: String,
    firstName: String,
    lastName: String,
    password: {
      type: String,
      get: (): undefined => undefined
    },
    createdAt: Date
  },
  {
    toJSON: {
      virtuals: true,
      getters: true
    }
  }
)

userSchema.virtual('fullName').get(function (this: {
  /** First name */
  firstName: string,
  /** Last name */
  lastName: string
}) {
  return `${this.firstName} ${this.lastName}`
})

export const userModel =
                mongoose.model<User & mongoose.Document>('User', userSchema)
