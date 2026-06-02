import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};
let _model = null;

const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1
}

// Normalize continous values (price, age) to 0-1 range
// Why? Keeps all features balanced so no one dominates training
// Formula: (val - min) / (max - min)
// Example price=129.99, minPrice=39.99, maxPrice=199.99 -> 0.56
const normalize = (val, min, max) => (val - min) / ((max - min) || 1)

function makeContext(products, users) {
    const ages = users.map(user => user.age)
    const prices = products.map(product => product.price)

    const maxAge = Math.max(...ages)
    const minAge = Math.min(...ages)

    const maxPrice = Math.max(...prices)
    const minPrice = Math.min(...prices)

    const colors = [...new Set(products.map(product => product.color))]
    const categories = [...new Set(products.map(product => product.category))]

    const colorsIndex = Object.fromEntries(
        colors.map((color, index) => {
            return [color, index]
        })
    )
    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => {
            return [category, index]
        })
    )

    // computar a média de idade dos compradores por produtos (ajuda a personalizar)
    const midAge = (minAge + maxAge) / 2
    const ageSums = {}
    const ageCounts = {}

    users.forEach(user => {
        user.purchases.forEach(purchase => {
            ageSums[purchase.name] = (ageSums[purchase.name] || 0) + user.age
            ageCounts[purchase.name] = (ageCounts[purchase.name] || 0) + 1
        })
    });

    const productAvgAgeNormalized = Object.fromEntries(
        products.map(product => {
            const avg = ageCounts[product.name] ?
                ageSums[product.name] / ageCounts[product.name] :
                midAge
                
                return [product.name, normalize(avg, minAge, maxAge)]
        })
    )
    
    return {
        products,
        users,
        colorsIndex,
        categoriesIndex,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numCategories: categories.length,
        numColors: colors.length,
        productAvgAgeNormalized,
        // price + age + categories + colors
        dimensions: 2 + categories.length + colors.length
    }
}

const oneHotWeighted = (index, length, weight) => 
    tf.oneHot(index, length).cast('float32').mul(weight)

function encodeProduct(product, context) {
    // normalizando dados para dicar de 0 a 1
    // e aplicar peso na recomendação
    const price = tf.tensor1d([
        normalize(
            product.price,
            context.minPrice,
            context.maxPrice,
        ) * WEIGHTS.price
    ])

    const age = tf.tensor1d([
        (
            context.productAvgAgeNormalized[product.name] ?? 0.5
        ) * WEIGHTS.age
    ])

    const category = oneHotWeighted(
        context.categoriesIndex[product.category],
        context.numCategories,
        WEIGHTS.category
    )

    const color = oneHotWeighted(
        context.colorsIndex[product.color],
        context.numColors,
        WEIGHTS.color
    )

    return tf.concat1d(
        [price, age, category, color]
    )
}

function encodeUser(user, context) {
    if(user.purchases.length) {
        return tf.stack(
            user.purchases.map(product => 
                encodeProduct(product, context)
            )
        )
        .mean(0)
        .reshape([
            1,
            context.dimensions
        ])
    }

    return tf.concat1d(
        [
            tf.zeros([1]), // Price is ignored
            tf.tensor1d([
                normalize(user.age, context.minAge, context.maxAge) 
                * WEIGHTS.age
            ]),
            tf.zeros([context.numCategories]), // Category is ignored
            tf.zeros([context.numColors]), // Colors is ignored
        ]
    ).reshape([1, context.dimensions])
}

function createTrainingData(context) {
    const inputs = []
    const labels = []

    context.users
        .filter(user => user.purchases.length)    
        .forEach(user => {
            const userVector = encodeUser(user, context).dataSync()
            context.products.forEach(product => {
                const productVector = encodeProduct(product, context).dataSync()

                const label = user.purchases.some(
                    purchase => purchase.name === product.name ?
                        1 :
                        0
                )

                //  combine user + product
                inputs.push([...userVector, ...productVector])
                labels.push(label)
            })
        })
    
    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        // tamanho = userVector + productVector
        inputDimension: context.dimensions * 2
    }
}

async function configureNeuralNetAndTrain(trainData) {
    const model = tf.sequential()
    
    /**
     * Entry layer
     * 
     * - inputShape: Number of features for instance of train
     * (trainData.inputDim)
     *  Example: If the product vector + user = 20 numbers, So the inputDim = 20
     * - Units: 128 Neurons (many "eyes"to detect the patterns)
     * - activation: 'relu' (keeps only positive signals, it helps to learn non-linears patterns)
     */
    model.add(
        tf.layers.dense({
            inputShape: [trainData.inputDimension],
            units: 128,
            activation: 'relu'
        })
    )
    /**
     * Hidden layer 1
     * - 64 neurons (less than the first layer: starting to compress the info)
     * - activation: 'relu' (still extracting relevant combinations of features)
     */
    model.add(
        tf.layers.dense({
            units: 64,
            activation: 'relu'
        })
    )
    /**
     * Hidden layer 2
     * - 32 neurons (more narrow again, destiling the most important informations)
     *  Example: From many signals, keeps only the strongest patterns
     * - activation: 'relu'
     */
    model.add(
        tf.layers.dense({
            units: 32,
            activation: 'relu'
        })
    )
    /**
     * Output Layer
     * - 1 Neuron because we'll return only one recommndation point
     * - activation: 'sigmoid'compress the result to 0-1 interval
     *  Example: 
     *  0.9 = strong recommendation
     *  0,1 = weak recommendation
     */ 
    model.add(
        tf.layers.dense({
            units: 1, 
            activation: 'sigmoid'
        })
    )

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
		metrics: ['accuracy']
    })

    await model.fit(
        trainData.xs,
        trainData.ys,
        {
            epochs: 100,
            batchSize: 32,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    postMessage({
                        type: workerEvents.trainingLog,
                        epoch,
                        loss: logs.loss,
                        accuracy: logs.acc
                    });
                }
            }
        }
    )

    return model
    
}

async function trainModel({ users }) {
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });
    const products = await(await fetch('/data/products.json')).json()

    const context = makeContext(products, users)
    context.productVectors = products.map(product => ({
        name: product.name,
        meta: {...product},
        vector: encodeProduct(product, context).dataSync()
    }))

    // It should be in a vector DB
    _globalCtx = context

    const trainData = createTrainingData(context)
    _model = await configureNeuralNetAndTrain(trainData)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });

}

function recommend(user, ctx) {
    if(!_model) return

    /**
     * Convert the provided user into a codified vectorized features
     * 
     *  (price ignored, normalized age, categories ignored, cores ignored)
     * 
     * It transform the user's info into the same numeric format that was used to train the model
     */
    const vetorizedUser = encodeUser(user, ctx).dataSync()

    /**
     * ⚠️‼️ For Real Apps
     * 
     * Store ALL product vectors in a vectorial DB (like Postgres (with vector plugin), Neo4j,
     * Pinecone, ChromaDB...)
     * 
     * Query to DB: Find the 200 closest products for the user's vector
     * 
     * Execute: _model.predict() ONLY on these products
     */


    /**
     * create pair of entries: for each product, concat the user's vector with
     * the product's vector
     * Why? the model predict the "compatibility score" for each pair (user, product)
     */
    const inputs = ctx.productVectors.map(({vector}) => {
        return [...vetorizedUser, ...vector]
    })

    /**
     * Converts all these pairs (user, product) into only one Tensor
     * Format: [numProducts, inputDim]
     */
    const inputTensor = tf.tensor2d(inputs)

    /**
     * Run the trained neural net against every pairs (user, product) at once
     * 
     * The result is a pontuation or each product between 0 and 1
     * As bigger it is bigger os the probability of the user want that product
     */
    const predictions = _model.predict(inputTensor)

    /**
     * Extract the scores to an regular JS array 
     */
    const scores = predictions.dataSync()
    
    const recommmendations = ctx.productVectors.map((item, index) => ({
        ...item.meta,
        name: item.name,
        score: scores[index], // model prediction for this product
    }))

    const sortedItems = recommmendations
        .sort((a,b) => b.score - a.score)

    /**
     * Send the poroducts recommendations sorted list to the main thread
     * (Ui to display them)
     */
    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations: sortedItems
    });
}

const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
