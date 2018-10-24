package main

import (
	"encoding/json"
	"errors"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/dynamodb"
	"github.com/aws/aws-sdk-go/service/dynamodb/dynamodbattribute"
	"github.com/aws/aws-sdk-go/service/dynamodb/expression"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"time"
	"strings"
)

type Storage interface {
	Init(path string) error
	HasKey(key string) bool
	ListKeys(prefix string) []string
	Save(key string, fields map[string]interface{}) error
	Load(key string) (map[string]interface{}, error)
	Delete(key string) error
}

//implement Storage interface
type DynamodbStorage struct {
	svc *dynamodb.DynamoDB
}

//implement Storage interface
type FileStorage struct {
	DataDir string
}

type S3Storage struct {
	BucketName string
	svc        *s3.S3
	downloader *s3manager.Downloader
	uploader   *s3manager.Uploader
	Region     string
    DataDir string
}

func (fs *FileStorage) Init(path string) error {
	fs.DataDir = path
	err := os.MkdirAll(fs.DataDir, 0777)
	return err
}

func (fs *FileStorage) HasKey(key string) bool {
	return Exists(path.Join(fs.DataDir, key+".json"))
}

func (fs *FileStorage) ListKeys(prefix string) []string {
	files, err := ioutil.ReadDir(path.Join(fs.DataDir, prefix))
	if err != nil {
		return []string{}
	}
	keys := []string{}
	for _, f := range files {
		key := f.Name()
		key = key[:len(key)-5]
		keys = append(keys, path.Join(prefix, key))
	}
	return keys
}

func (fs *FileStorage) Save(key string, fields map[string]interface{}) error {
	os.MkdirAll(path.Join(fs.DataDir, filepath.Dir(key)), 0777)
	path := path.Join(fs.DataDir, key+".json")
	Info.Println(path)
	json, err := json.MarshalIndent(fields, "", "  ")
	if err != nil {
		return err
	}
	err = ioutil.WriteFile(path, json, 0644)
	if err != nil {
		return err
	} else {
		Info.Println("Saving file of", key)
	}
	return nil
}

func (fs *FileStorage) Load(key string) (map[string]interface{}, error) {
	var fields map[string]interface{}
	projectFilePath := path.Join(fs.DataDir, key+".json")
	// TODO: check whether the file exists first
	projectFileContents, err := ioutil.ReadFile(projectFilePath)
	if err != nil {
		return fields, &NotExistError{projectFilePath}
	}
	err = json.Unmarshal(projectFileContents, &fields)
	if err != nil {
		return fields, err
	}
	return fields, nil
}

func (fs *FileStorage) Delete(key string) error {
	ItemDir := path.Join(fs.DataDir, key)
	err := os.RemoveAll(ItemDir)
	return err
}

func (ds *DynamodbStorage) Init(path string) error {
	sess, err := session.NewSession(&aws.Config{
		Region: aws.String(path)},
	)
	// Create DynamoDB client
	ds.svc = dynamodb.New(sess)
	if err != nil {
		return err
	}
	if !ds.HasTable() {
		Info.Println("Creating scalabel table")
		input := &dynamodb.CreateTableInput{
			AttributeDefinitions: []*dynamodb.AttributeDefinition{
				{
					AttributeName: aws.String("Key"),
					AttributeType: aws.String("S"),
				},
			},
			KeySchema: []*dynamodb.KeySchemaElement{
				{
					AttributeName: aws.String("Key"),
					KeyType:       aws.String("HASH"),
				},
			},
			ProvisionedThroughput: &dynamodb.ProvisionedThroughput{
				ReadCapacityUnits:  aws.Int64(10),
				WriteCapacityUnits: aws.Int64(10),
			},
			TableName: aws.String("scalabel"),
		}
		ds.svc.CreateTable(input)
		for i := 0; i < 30; i++ {
			if ds.HasTable() {
				return nil
			}
			time.Sleep(1 * time.Second)
		}
		return errors.New("Cannot find scalabel table after 30s")
	}
	return nil
}

func (ds *DynamodbStorage) HasKey(key string) bool {
	result, err := ds.svc.GetItem(&dynamodb.GetItemInput{
		TableName: aws.String("scalabel"),
		Key: map[string]*dynamodb.AttributeValue{
			"Key": {
				S: aws.String(key),
			},
		},
	})
	if (err != nil) || (len(result.Item) == 0) {
		return false
	}
	return true
}

func (ds *DynamodbStorage) ListKeys(prefix string) []string {
	filt := expression.BeginsWith(expression.Name("Key"), prefix)
	proj := expression.NamesList(expression.Name("Key"))
	expr, err := expression.NewBuilder().WithFilter(filt).WithProjection(proj).Build()
	if err != nil {
		Error.Println(err)
	}
	params := &dynamodb.ScanInput{
		ExpressionAttributeNames:  expr.Names(),
		ExpressionAttributeValues: expr.Values(),
		FilterExpression:          expr.Filter(),
		ProjectionExpression:      expr.Projection(),
		TableName:                 aws.String("scalabel"),
	}
	// Make the DynamoDB Query API call
	result, err := ds.svc.Scan(params)
	if err != nil {
		Error.Println(err)
	}
	keys := []string{}
	for _, i := range result.Items {
		var fields map[string]string
		err = dynamodbattribute.UnmarshalMap(i, &fields)
		keys = append(keys, fields["Key"])
	}
	if err != nil {
		Error.Println(err)
	}
	return keys
}

func (ds *DynamodbStorage) Save(key string, fields map[string]interface{}) error {
	fields["Key"] = key
	av, err := dynamodbattribute.MarshalMap(fields)
	input := &dynamodb.PutItemInput{
		Item:      av,
		TableName: aws.String("scalabel"),
	}
	_, err = ds.svc.PutItem(input)
	if err != nil {
		return err
	}
	Info.Println("Successfully added an item to the table on dynamodb")
	return nil
}

func (ds *DynamodbStorage) Load(key string) (map[string]interface{}, error) {
	result, err := ds.svc.GetItem(&dynamodb.GetItemInput{
		TableName: aws.String("scalabel"),
		Key: map[string]*dynamodb.AttributeValue{
			"Key": {
				S: aws.String(key),
			},
		},
	})
	var fields map[string]interface{}
	if err != nil {
		return fields, &NotExistError{key}
	}
	err = dynamodbattribute.UnmarshalMap(result.Item, &fields)
	if err != nil {
		return fields, err
	}
	delete(fields, "Key")
	return fields, nil
}

func (ds *DynamodbStorage) Delete(key string) error {
	input := &dynamodb.DeleteItemInput{
		Key: map[string]*dynamodb.AttributeValue{
			"Key": {
				S: aws.String(key),
			},
		},
		TableName: aws.String("scalabel"),
	}
	_, err := ds.svc.DeleteItem(input)
	if err != nil {
		return err
	}
	return nil
}

// Check whether scalabel table already exists
func (ds *DynamodbStorage) HasTable() bool {
	input := &dynamodb.DescribeTableInput{
		TableName: aws.String("scalabel"),
	}
	_, err := ds.svc.DescribeTable(input)
	if err != nil {
		return false
	}
	return true
}

func (ss *S3Storage) Init(path string) error {
    info := strings.Split(path, ":")
    ss.Region = info[0]
    bucketPath := strings.Split(info[1], "/")
    ss.BucketName = bucketPath[0]
    ss.DataDir = strings.Join(bucketPath[1:], "/")
	sess, err := session.NewSession(&aws.Config{
		Region: aws.String(ss.Region)},
	)
	ss.downloader = s3manager.NewDownloader(sess)
	ss.uploader = s3manager.NewUploader(sess)
	// Create S3 client
	ss.svc = s3.New(sess)
	if err != nil {
		return err
	}
	if !ss.HasBucket() {
		Info.Println("Creating scalabel s3 bucket")
		_, err = ss.svc.CreateBucket(&s3.CreateBucketInput{
			Bucket: aws.String(ss.BucketName),
		})
		if err != nil {
			return err
		}
		// Wait until bucket is created before finishing
		Info.Println("Waiting for bucket scalabel to be created...")
		err = ss.svc.WaitUntilBucketExists(&s3.HeadBucketInput{
			Bucket: aws.String(ss.BucketName),
		})
		if err != nil {
			return err
		}
	}
	return nil
}

func (ss *S3Storage) HasKey(key string) bool {
	input := &s3.GetObjectInput{
		Bucket: aws.String(ss.BucketName),
		Key:    aws.String(path.Join(ss.DataDir, key)),
	}
	_, err := ss.svc.GetObject(input)
	if err != nil {
		return false
	}
	return true
}

func (ss *S3Storage) ListKeys(prefix string) []string {
	params := &s3.ListObjectsInput{
		Bucket: aws.String(ss.BucketName),
		Prefix: aws.String(path.Join(ss.DataDir,prefix)),
	}
	resp, err := ss.svc.ListObjects(params)
	if err != nil {
		Error.Println(err)
	}
	keys := []string{}
	for _, key := range resp.Contents {
		keys = append(keys, *key.Key)
	}
	return keys
}

func (ss *S3Storage) Save(key string, fields map[string]interface{}) error {
	tmpfile, err := ioutil.TempFile("", "*.json")
	if err != nil {
		return err
	}
	defer os.Remove(tmpfile.Name())
	json, err := json.MarshalIndent(fields, "", "  ")
	if err != nil {
		return err
	}
	err = ioutil.WriteFile(tmpfile.Name(), json, 0644)
	if err != nil {
		return err
	}
	_, err = ss.uploader.Upload(&s3manager.UploadInput{
		Bucket: aws.String(ss.BucketName),
		Key:    aws.String(path.Join(ss.DataDir,key)),
		Body:   tmpfile,
	})
	if err != nil {
		return err
	} else {
		Info.Println("Successfully added an item to the S3")
	}
	return nil
}

func (ss *S3Storage) Load(key string) (map[string]interface{}, error) {
	var fields map[string]interface{}
	if !strings.HasPrefix(key, ss.DataDir) {
        key = path.Join(ss.DataDir, key)
    }
	tmpfile, err := ioutil.TempFile("", "*.json")
	if err != nil {
		return fields, err
	}
	defer os.Remove(tmpfile.Name())
	_, err = ss.downloader.Download(tmpfile,
		&s3.GetObjectInput{
			Bucket: aws.String(ss.BucketName),
			Key:    aws.String(key),
		})
	if err != nil {
		return fields, &NotExistError{key}
	}
	projectFileContents, err := ioutil.ReadFile(tmpfile.Name())
	if err != nil {
		return fields, &NotExistError{tmpfile.Name()}
	}
	err = json.Unmarshal(projectFileContents, &fields)
	if err != nil {
		return fields, err
	}
	return fields, nil
}

func (ss *S3Storage) Delete(key string) error {
	_, err := ss.svc.DeleteObject(&s3.DeleteObjectInput{Bucket: aws.String(ss.BucketName), Key: aws.String(key)})
	if err != nil {
		return err
	}
	err = ss.svc.WaitUntilObjectNotExists(&s3.HeadObjectInput{
		Bucket: aws.String(ss.BucketName),
		Key:    aws.String(key),
	})
	if err != nil {
		return err
	}
	return nil
}

// Check whether scalabel table already exists
func (ss *S3Storage) HasBucket() bool {
	input := &s3.HeadBucketInput{
		Bucket: aws.String(ss.BucketName),
	}
	_, err := ss.svc.HeadBucket(input)
	if err != nil {
		return false
	}
	return true
}
