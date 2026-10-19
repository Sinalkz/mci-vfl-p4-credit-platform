// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import {TalmudLib} from "../TalmudDivision.sol";

/// @notice Measurement helper: execution gas of TalmudLib.divide when called
/// internally, as a settlement contract would. Excludes the 21000 base cost,
/// calldata, ABI decoding and the external-call overhead.
contract TalmudGasProbe {
    function measure(uint256 estate, uint256[] memory claims)
        external
        view
        returns (uint256 gasUsed, uint256[] memory awards)
    {
        uint256 before = gasleft();
        awards = TalmudLib.divide(estate, claims);
        gasUsed = before - gasleft();
    }
}
