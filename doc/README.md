# Build the Doc

```bash
cd doc/
git clone git@github.com:scalabel/scalabel-doc-media.git media
make html
```

Update the doc on the server:

```bash
sh upload.sh
```

The doc is hosted at https://doc.scalabel.ai.

The AWS S3 url: https://s3-us-west-2.amazonaws.com/doc.scalabel.ai/index.html
