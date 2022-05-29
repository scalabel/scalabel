import express from "express"
import request from "supertest"
import { parseMultipartFormData } from "../../src/server/middleware/multipart"

describe("test middleware", () => {
  test("parsing multipart/form-data", (done) => {
    const endpoint = "/not/important"

    const app = express()
    app.post(endpoint, (req, res) => {
      parseMultipartFormData(req)
        .then(({ fields, files }) => {
          expect(fields).toMatchObject({ foo: "bar", hello: "world" })
          expect(files.file1).toHaveProperty("filepath")
          expect(files.file2).toHaveProperty("filepath")
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
