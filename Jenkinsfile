pipeline{
    agent any

    stages{
        stage("Checkout"){
            steps{
                checkout([$class: 'GitSCM', branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'ef70b0d3-000a-4c42-a98f-d2c045e33251', url: 'https://github.com/jayaharisai5/jenkins.git']]])
            }
        }
        stage("build"){
            steps{
                git branch: 'main', credentialsId: 'ef70b0d3-000a-4c42-a98f-d2c045e33251', url: 'https://github.com/jayaharisai5/jenkins.git'
            }
        }
        stage("load_data"){
            steps{
                sh 'python3 load_data.py'
            }
        }
        stage("feature_engineering"){
            steps{
                sh 'python3 data_analysis.py'
            }
        }
        stage("pre_processing"){
            steps{
                sh 'python3 feature_engineering.py'
            }
        }
        stage("model_selection"){
            steps{
                sh 'python3 data_preprocessing.py'
            }
        }
        stage("model_selection"){
            steps{
                sh 'python3 model_selection.py'
            }
        }
        
    }
    post{
        always{
            emailext body:"summery", subject: "Pipeline Status", to: 'jayaharisai1212@gmail.com'
        }
    }
}