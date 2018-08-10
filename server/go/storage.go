package main

import (
	"encoding/json"
	"errors"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/dynamodb"
	"github.com/aws/aws-sdk-go/service/dynamodb/dynamodbattribute"
	"github.com/aws/aws-sdk-go/service/dynamodb/expression"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"time"
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
	projectFileContents, err := ioutil.ReadFile(projectFilePath)
	if err != nil {
		return fields, err
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
		return fields, err
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
