import express from "express"
import request from "supertest"
import { File } from "formidable"
import { parseMultipartFormData } from "../../src/server/middleware/multipart"

describe("test middleware", () => {
  test("parsing multipart/form-data", (done) => {
    const endpoint = "/not/important"

    const app = express()
    app.post(endpoint, (req, res) => {
      parseMultipartFormData(req)
        .then(({ fields, files }) => {
          expect(fields).toMatchObject({ foo: "bar", hello: "world" })
          expectSingleFile(files.file1)
          expectSingleFile(files.file2)
          res.sendStatus(200)
        })
        .catch(() => res.sendStatus(500))
    })

    request(app)
      .post(endpoint)
      .field("foo", "bar")
      .field("hello", "world")
      .attach("file1", Buffer.from("content1"), "filename1")
      .attach("file2", Buffer.from("content2"), "filename2")
      .expect(200)
      .then(() => done())
      .catch(done)
  })
})

/**
 * Expect that a single valid `File` is provided.
 *
 * @param f - the parsed file(s)
 */
function expectSingleFile(f: File | File[]): void {
  expect(f).toBeInstanceOf(File)

  // formidable once changed its public API from using `path` to `filepath`.
  // See https://github.com/scalabel/scalabel/pull/449/files.
  // To avoid unexpected breaks in the future, we check if `filepath` is valid.
  expect((f as File).filepath).toBeTruthy()
}
