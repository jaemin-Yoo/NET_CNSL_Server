<?php
header("Content-Type: text/html; charset=UTF-8");
$conn= new mysqli("127.0.0.1", "root", "4556", "location");
mysqli_query($conn,'SET NAMES utf8');

$lat= $_POST['Latitude'];
$lng= $_POST['Longitude'];

$lat_double= (double)$lat;
$lng_double= (double)$lng;


$sql_cctv= "select * from cctv where Latitude<$lat_double and Latitude>$lat_double and Longitude<$lng_double and Longitude>$lng_double";
$sql_person= "select * from person where Latitude<$lat_double and Latitude>$lat_double and Longitude<$lng_double and Longitude>$lng_double";

$res_cctv= $conn->query($sql_cctv);
$res_person= $conn->query($sql_person);

$response= array();
$response["success"]= false;

$num=0;

if(mysqli_fetch_array($res_cctv)!=NULL) {
    if(mysqli_fetch_array($res_person)!=NULL) {
        while($row=mysqli_fetch_array($res_person)) {
            $num++;
            $response["number"]= $num;
            $numstr= (string)$num;
            $response["Latitude$numstr"]= $row['Latitude'];
            $response["Longitude$numstr"]= $row['Longitude'];
        }
    }
    $response["success"]= true;
}

echo json_encode($response);
?>