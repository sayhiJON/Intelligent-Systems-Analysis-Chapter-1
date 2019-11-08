open System.IO

let countWords (text:string) : int =
    let count = text.Split().Length
    let stream = File.OpenWrite "test.txt"
    let writer = new StreamWriter (stream)
    writer.WriteLine text
    writer.WriteLine (count.ToString())
    writer.Close()

    count
        
countWords "hello, world";;