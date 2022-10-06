/*
 * SPDX-License-Identifier: Apache-2.0
 */

'use strict';

const { FileSystemWallet, Gateway, X509WalletMixin } = require('fabric-network');
const fs = require('fs');
const path = require('path');

const ccpPath = path.resolve(__dirname, '..', 'pbft-network-10', 'connection-org1.json');

async function main() {
    try {
        let user = 'user3'
        // Create a new file system based wallet for managing identities.
        const walletPath = path.join(process.cwd(), 'wallet');
        const wallet = new FileSystemWallet(walletPath);
        console.log(`Wallet path: ${walletPath}`);
        
        // Check to see if we've already enrolled the user.
        const userExists = await wallet.exists(user);
        let dir = walletPath + '/' + user;
        if (userExists) {
            try {
                console.log("dir: ", dir);
                fs.rmdirSync(dir, { recursive: true });
                console.log('An identity for the user "user3" already exists in the wallet, delete and create new...');
                console.log(`${dir} is deleted!`);
            } catch (err) {
                console.error(`Error while deleting ${dir}.`);
            }
        }
        
        // Check to see if we've already enrolled the admin user.
        const adminExists = await wallet.exists('admin');
        if (!adminExists) {
            console.log('An identity for the admin user "admin" does not exist in the wallet');
            console.log('Run the enrollAdmin.js application before retrying');
            return;
        }
        
        // Create a new gateway for connecting to our peer node.
        const gateway = new Gateway();
        await gateway.connect(ccpPath, { wallet, identity: 'admin', discovery: { enabled: true, asLocalhost: true } });

        // Get the CA client object from the gateway for interacting with the CA.
        const ca = gateway.getClient().getCertificateAuthority();
        const adminIdentity = gateway.getCurrentIdentity();
        
        // Register the user, enroll the user, and import the new identity into the wallet.
        const secret = await ca.register({ affiliation: 'org1.department1', enrollmentID: user, role: 'client' }, adminIdentity);
        const enrollment = await ca.enroll({ enrollmentID: user, enrollmentSecret: secret });
        const userIdentity = X509WalletMixin.createIdentity('Org1MSP', enrollment.certificate, enrollment.key.toBytes());
        await wallet.import(user, userIdentity);;
        console.log('Successfully registered and enrolled admin user "user3" and imported it into the wallet');

    } catch (error) {
        console.error(`Failed to register user "user3": ${error}`);
        process.exit(1);
    }
}

main();
