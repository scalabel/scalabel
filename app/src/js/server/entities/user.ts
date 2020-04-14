import * as mongoose from 'mongoose'

export interface User {
  /** Mongo id */
  _id: string
  /** Given name */
  givenName: string
  /** Family name */
  familyName: string
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
    givenName: String,
    familyName: String,
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
  /** Given name */
  givenName: string,
  /** Family name */
  familyName: string
}) {
  return `${this.givenName} ${this.familyName}`
})

export const userModel =
                mongoose.model<User & mongoose.Document>('User', userSchema)
