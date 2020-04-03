import * as mongoose from 'mongoose'

export interface Reset {
  /** Mongo id */
  _id: string
  /** User id */
  userId: string
  /** Reset token */
  token: string
  /** Applied */
  applied: boolean
  /** Register time */
  createdAt: Date
}

const resetSchema = new mongoose.Schema(
  {
    userId: String,
    token: String,
    applied: Boolean,
    createdAt: Date
  },
  {
    toJSON: {
      getters: true
    }
  }
)

export const resetModel =
                mongoose.model<Reset & mongoose.Document>('Reset', resetSchema)
