package main

import (
  "fmt"
  "log"
  "encoding/csv"
  "io"
  "os"
  "strconv"
  "math"
  )

const (
  n=699
)

var a_0[]float64
var a_1[]float64

func datasetSetUp(filePath string, x_slice []float64, y_slice []float64) ([]float64, []float64){
  //import csv file and parse the content
  train_csv,err := os.Open(filePath);
  if err != nil{
    log.Fatalln("file open failed with error: ",err);
  }
  //fmt.Println(train_csv);
  reader := csv.NewReader(train_csv);

  for{
    datapoint,err := reader.Read()
    if err == io.EOF{
      break
    }
    if err != nil{
      log.Fatalln("file read record failed: ",err);
    }
    //fmt.Printf("x:  %s  y:  %s\n",datapoint[0],datapoint[1])
    x,_ := strconv.ParseFloat(datapoint[0],64)
    y,_ := strconv.ParseFloat(datapoint[1],64)
    x_slice = append(x_slice,x)
    y_slice = append(y_slice,y)

  }
  return x_slice,y_slice
}

func LinearRegression(x_train []float64, y_train []float64){
  //n:= 700
  var mean_sq_error float64
  var np_sum float64
  var np_sum_error float64
  var y[]float64
  var error[]float64
  alpha:= 0.0001

  epochs := 0
  a_0 = make([]float64,len(x_train))
  a_1 = make([]float64,len(x_train))
  y = make([]float64,len(x_train))
  error = make([]float64,len(x_train))

  for epochs<1000{
    for j,_ := range x_train{
      y[j]=a_0[j]+a_1[j]*x_train[j]
      error[j]= y[j]-y_train[j]
    }


    for i:=0;i<n;i++ {
      mean_sq_error += math.Pow(error[i],2)
      np_sum += error[i]
      np_sum_error += error[i]*x_train[i]
    }
    mean_sq_error = mean_sq_error/n
    for i:=0;i<n;i++ {
      a_0[i] = a_0[i] - alpha * 2 * np_sum/n
      a_1[i] = a_1[i] - alpha * 2 * np_sum_error/n
    }
    epochs += 1
    if(epochs%10 == 0){
      fmt.Println(mean_sq_error)
    }

  }

}

func predict(x_test []float64, y_test []float64){
  var y_pred[]float64
  var explained_variance float64
  var avg_actual float64
  var total_variance float64
  var r2 float64
  y_pred = make([]float64,len(y_test))
  for i:=0;i<len(x_test);i++{
    y_pred[i] = a_0[i] + a_1[i] * x_test[i]
  }
  for i:=0;i<len(y_test);i++{
    explained_variance += math.Pow((y_pred[i]-y_test[i]),2)
  }
  for i:=0;i<len(y_test);i++{
    avg_actual += y_test[i]
  }
  avg_actual = avg_actual/float64(len(y_test))
  for i:=0;i<len(y_pred);i++{
    total_variance += math.Pow((y_pred[i]-avg_actual),2)
  }
  r2 = 1-(explained_variance/total_variance)
  fmt.Printf("R2: %f",r2)
}

func main() {
  var x_train []float64
  var y_train []float64
  var x_test []float64
  var y_test []float64
  filePath_train := "dataset/LinearRegression/train.csv"
  x_train, y_train = datasetSetUp(filePath_train, x_train, y_train)

  filePath_test := "dataset/LinearRegression/test.csv"
  x_test, y_test = datasetSetUp(filePath_test, x_test, y_test)

  //for _,element:=range y_test{
  //  fmt.Printf("y:%f  ",element)
  //}
  LinearRegression(x_train,y_train)
  predict(x_test,y_test)
}
