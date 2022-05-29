import { IncomingMessage } from "http"
import { NextFunction, Request, Response } from "express"
import formidable, { Fields, Files } from "formidable"

const maxFileSize = 1000 * 1024 * 1024 // 1G

const multipart = (req: Request, _: Response, next: NextFunction): void => {
  parseMultipartFormData(req)
    .then(({ fields, files }) => {
      Object.assign(req, { fields, files })
      next()
    })
    .catch(next)
}

/**
 * Parse an incoming multipart/form-data request.
 *
 * @param req
 */
export async function parseMultipartFormData(
  req: IncomingMessage
): Promise<{ fields: Fields; files: Files }> {
  return await new Promise((resolve, reject) => {
    const form = new formidable.IncomingForm({ maxFileSize: maxFileSize })
    form.parse(req, (err, fields, files) => {
      // `err` is defined as `any` by formidable.
      // eslint-disable-next-line  @typescript-eslint/strict-boolean-expressions
      if (err) {
        reject(err)
      } else {
        resolve({ fields, files })
      }
    })
  })
}

export default multipart
